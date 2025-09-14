# src/retinaNet.py

import torch
import os
import json
import random
import math
import argparse
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageFile, ImageDraw, ImageFont

# PyTorch 관련 라이브러리
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor
from torch.amp import autocast, GradScaler

# torchvision 버전에 따라 호환되는 RetinaNet을 자동으로 불러옵니다.
try:
    from torchvision.models.detection import retinanet_resnet50_fpn_v2 as retinanet_factory
except ImportError:
    from torchvision.models.detection import retinanet_resnet50_fpn as retinanet_factory

# pycocotools는 평가 시에만 필요
try:
    from pycocotools.coco import COCO as COCOapi
    from pycocotools.cocoeval import COCOeval
except ImportError:
    print("Warning: pycocotools not found. Please install it (`pip install pycocotools`) to run evaluation.")
    COCOapi, COCOeval = None, None

ImageFile.LOAD_TRUNCATED_IMAGES = True

# -----------------------------------------------------------------------------
# 데이터 처리 관련 함수 및 클래스
# -----------------------------------------------------------------------------

def load_or_merge_coco(ann_dir, merged_path, force_rebuild=False):
    """
    통합 COCO JSON 파일을 로드하거나, 없을 경우 새로 생성합니다.

    Arg:
        ann_dir (Path): 원본 COCO JSON 파일들이 담긴 폴더 경로.
        merged_path (Path): 통합된 JSON 파일을 저장하거나 읽어올 경로.
        force_rebuild (bool): True일 경우, 기존 통합 파일이 있어도 강제로 다시 병합.

    Return:
        (dict): 통합된 COCO 데이터 딕셔너리.

    Description:
        여러 개로 나뉜 COCO Annotation 파일들을 하나의 파일로 병합합니다.
        병합된 파일이 이미 존재하면 해당 파일을 바로 로드하여 시간을 절약합니다.
        병합 과정에서 모든 이미지와 Annotation에 고유한 ID를 새로 부여합니다.
    """
    if merged_path.exists() and not force_rebuild:
        print(f"[INFO] 캐시된 통합 Annotation 파일 사용: {merged_path}")
        with open(merged_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    print(f"[INFO] 통합 Annotation 파일({merged_path})이 없어 새로 생성합니다.")
    json_files = [p for p in ann_dir.rglob("*.json")]
    assert json_files, f"Annotation 파일을 찾을 수 없습니다: {ann_dir}"

    new_images, new_annotations, category_names = [], [], set()
    next_img_id, next_ann_id = 1, 1

    for json_path in json_files:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"경고: {json_path} 파일 읽기 실패. 건너뜁니다. ({e})")
            continue
        
        old_to_new_img_id = {}
        for img in data.get('images', []):
            new_img = {'id': next_img_id, 'file_name': Path(img['file_name']).name, 'width': img['width'], 'height': img['height']}
            new_images.append(new_img)
            old_to_new_img_id[img['id']] = next_img_id
            next_img_id += 1

        local_cats = {c['id']: c['name'] for c in data.get('categories', [])}
        for name in local_cats.values():
            category_names.add(name)

        for ann in data.get('annotations', []):
            if 'bbox' not in ann or ann['image_id'] not in old_to_new_img_id:
                continue
            x, y, w, h = ann['bbox']
            if not (w > 0 and h > 0): continue
            
            new_ann = {'id': next_ann_id, 'image_id': old_to_new_img_id[ann['image_id']], 'category_name': local_cats.get(ann['category_id']), 'bbox': ann['bbox'], 'iscrowd': ann.get('iscrowd', 0), 'area': w * h}
            new_annotations.append(new_ann)
            next_ann_id += 1
            
    sorted_names = sorted(list(category_names))
    name_to_cat_id = {name: i + 1 for i, name in enumerate(sorted_names)}
    new_categories = [{'id': cat_id, 'name': name} for name, cat_id in name_to_cat_id.items()]
    
    for ann in new_annotations:
        ann['category_id'] = name_to_cat_id[ann['category_name']]
        del ann['category_name']
    
    COCO = {"images": new_images, "annotations": new_annotations, "categories": new_categories}

    merged_path.parent.mkdir(exist_ok=True, parents=True)
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(COCO, f, ensure_ascii=False, indent=2)
    
    print(f"[SUCCESS] 파일 병합 완료 및 저장: {merged_path}")
    return COCO

class CocoLikeDetection(Dataset):
    """
    객체 탐지 모델 학습을 위한 PyTorch Dataset 클래스.

    Description:
        주어진 이미지 ID 리스트를 바탕으로 이미지를 로드하고, Annotation 정보를 텐서(Tensor) 형태로 변환합니다.
        학습 시(augment=True) 간단한 데이터 증강(Augmentation)을 포함합니다.
    """
    def __init__(self, image_ids, img_dir, id2path, ann_by_img, augment=True):
        """
        Arg:
            image_ids (list): 학습 또는 검증에 사용할 이미지 ID의 리스트.
            img_dir (Path): 원본 이미지 파일들이 담긴 폴더 경로.
            id2path (dict): 이미지 ID를 실제 파일 경로로 매핑하는 딕셔너리.
            ann_by_img (dict): 이미지 ID를 해당 이미지의 Annotation 리스트로 매핑하는 딕셔너리.
            augment (bool): 데이터 증강을 적용할지 여부.
        """
        self.ids = image_ids
        self.img_dir = img_dir
        self.id2path = id2path
        self.ann_by_img = ann_by_img
        self.augment = augment

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        im_id = self.ids[idx]
        with Image.open(self.id2path[im_id]) as im:
            im = im.convert("RGB")
            W, H = im.size

        anns = self.ann_by_img.get(im_id, [])
        boxes_xyxy, labels = [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            x1, y1, x2, y2 = max(0.0, x), max(0.0, y), min(float(W), x + w), min(float(H), y + h)
            if x2 > x1 and y2 > y1:
                boxes_xyxy.append([x1, y1, x2, y2])
                labels.append(int(a["category_id"]))

        boxes = torch.tensor(boxes_xyxy, dtype=torch.float32) if boxes_xyxy else torch.zeros((0, 4))
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,))
        
        if labels.numel():
            labels = labels - 1 # RetinaNet은 0-based label을 사용 (0 ~ K-1)
            
        im, boxes = self._resize_keep_aspect(im, boxes)
        if self.augment:
            if random.random() < 0.5:
                im = TF.hflip(im)
                boxes = self._hflip_boxes_xyxy(boxes, im.size[0])
            im = TF.adjust_brightness(im, 0.9 + 0.2 * random.random())
            im = TF.adjust_contrast(im, 0.9 + 0.2 * random.random())
        
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([im_id])}
        return to_tensor(im), target

    def _resize_keep_aspect(self, img, boxes, short=800, max_sz=1333):
        W,H=img.size; s,l=(H,W) if H<W else (W,H); sc=short/max(1,s)
        if round(l*sc)>max_sz: sc=max_sz/max(1,l)
        if abs(sc-1.0)<1e-6: return img, boxes
        nW,nH=int(round(W*sc)), int(round(H*sc)); img=img.resize((nW,nH),Image.BILINEAR)
        if boxes.numel()>0: boxes=boxes*sc
        return img, boxes

    def _hflip_boxes_xyxy(self, boxes, w):
        if boxes.numel()==0: return boxes
        x1=boxes[:,0].clone(); x2=boxes[:,2].clone(); boxes[:,0]=w-x2; boxes[:,2]=w-x1
        return boxes

def collate(batch):
    """
    DataLoader를 위한 커스텀 collate 함수.

    Arg:
        batch: Dataset의 __getitem__에서 반환된 (이미지, 타겟) 튜플의 리스트.
    
    Description:
        PyTorch DataLoader는 배치(batch) 단위로 데이터를 묶을 때 이 함수를 사용합니다.
        이미지와 타겟을 각각의 리스트로 분리하여 반환합니다.
    """
    return list(zip(*batch))


# -----------------------------------------------------------------------------
# 학습 및 평가 관련 함수
# -----------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scaler, device, K):
    """
    한 에폭(epoch) 동안 모델을 학습합니다.

    Arg:
        model: 학습할 PyTorch 모델.
        loader (DataLoader): 학습 데이터 로더.
        optimizer: 최적화 함수 (e.g., SGD, AdamW).
        scaler (GradScaler): AMP(혼합 정밀도) 사용을 위한 스케일러.
        device: 학습에 사용할 장치 (e.g., 'cuda', 'cpu').
        K (int): 전체 클래스 수.

    Return:
        (float): 해당 에폭의 평균 학습 손실(loss).
    """
    model.train()
    total_loss = 0
    for imgs, targets in loader:
        imgs = [im.to(device) for im in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        for t in targets:
            t['labels'] = t['labels'].to(torch.long)

        optimizer.zero_grad()
        with autocast('cuda', enabled=True):
            loss_dict = model(imgs, targets)
            loss = sum(l for l in loss_dict.values())
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def validate_loss(model, loader, device, K):
    """
    한 에폭(epoch) 동안 모델을 검증합니다.

    Arg:
        (train_one_epoch과 동일)

    Return:
        (float): 해당 에폭의 평균 검증 손실(loss).
    
    Description:
        torch.no_grad()를 사용하여 gradient 계산을 비활성화하고,
        모델의 가중치를 업데이트하지 않고 순수하게 성능(loss)만 측정합니다.
    """
    model.train() # torchvision detection 모델은 loss 계산을 위해 train() 모드여야 함
    total_loss = 0
    for imgs, targets in loader:
        imgs = [im.to(device) for im in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        for t in targets:
            t['labels'] = t['labels'].to(torch.long)
        with autocast('cuda', enabled=True):
            loss_dict = model(imgs, targets)
            loss = sum(l for l in loss_dict.values())
        total_loss += loss.item()
    model.eval() # 다시 평가 모드로 전환
    return total_loss / len(loader)

def run_evaluation(model, val_ids, device, id2path, ann_by_img, COCO):
    """
    학습된 모델의 mAP 성능을 측정합니다.

    Description:
        (향후 구현) pycocotools를 사용하여 mAP@[0.75:0.95] 점수를 계산하고 출력합니다.
    """
    print("Warning: run_evaluation 함수는 아직 구현되지 않았습니다.")
    pass

def run_visualization(model, vis_ids, device, id2path, cat_id2name, vis_dir):
    """
    모델의 예측 결과를 이미지에 그려서 시각화합니다.

    Description:
        (향후 구현) 검증 데이터셋의 일부 이미지에 대한 모델의 예측 결과를
        Bbox와 함께 그려서 지정된 폴더에 저장합니다.
    """
    print("Warning: run_visualization 함수는 아직 구현되지 않았습니다.")
    pass


# -----------------------------------------------------------------------------
# 메인 실행 함수
# -----------------------------------------------------------------------------

def main(args):
    """
    전체 학습 파이프라인을 실행하는 메인 함수.

    Arg:
        args (argparse.Namespace): 스크립트 실행 시 전달된 커맨드 라인 인자.

    Description:
        데이터 로드, 모델 생성, 학습 루프 실행 등 전체 과정을 총괄합니다.
        스크립트 실행 시 전달된 인자(args)를 바탕으로 모든 설정을 구성합니다.
    """
    # 1. 경로 및 하이퍼파라미터 설정
    ROOT = Path(args.data_root)
    IMG_DIR = ROOT / "train_images"
    ANN_DIR = ROOT / "train_annotations"
    OUTPUT_DIR = ROOT / "outputs"
    SPLITS_PATH = OUTPUT_DIR / "RetinaNet_splits.json"
    MERGED_JSON = OUTPUT_DIR / "RetinaNet_coco_merged.json"
    CKPT_DIR = ROOT / "checkpoints"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    SEED = 42
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = 2
    EPOCHS = args.epochs
    LR = args.lr
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_AMP = True

    random.seed(SEED)
    torch.manual_seed(SEED)
    print(f"[INFO] 프로젝트 경로: {ROOT}")
    print(f"[INFO] 학습 장치: {DEVICE}")

    # 2. 데이터 로드/병합
    COCO = load_or_merge_coco(ANN_DIR, MERGED_JSON)
    
    # 3. 인덱싱 및 데이터 분할
    img_meta = {im["id"]: im for im in COCO["images"]}
    ann_by_img = defaultdict(list)
    for a in COCO["annotations"]:
        ann_by_img[a["image_id"]].append(a)
    id2path = {i: (IMG_DIR / Path(img_meta[i]["file_name"]).name) for i in img_meta}
    K = len(COCO["categories"])
    
    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)
    train_ids = [i for i in splits['train_ids'] if id2path.get(i) and id2path[i].exists()]
    val_ids = [i for i in splits['val_ids'] if id2path.get(i) and id2path[i].exists()]
    print(f"[INFO] 데이터 분할 완료: Train {len(train_ids)}개 / Validation {len(val_ids)}개")

    # 4. 데이터셋 및 데이터로더 생성
    train_ds = CocoLikeDetection(train_ids, IMG_DIR, id2path, ann_by_img, augment=True)
    val_ds   = CocoLikeDetection(val_ids, IMG_DIR, id2path, ann_by_img, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate)
    
    # 5. 모델, 옵티마이저, 스케줄러 준비
    model = retinanet_factory(weights_backbone="DEFAULT", num_classes=K)
    model.to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=1e-4)
    lr_sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    scaler = GradScaler('cuda', enabled=USE_AMP)
    print(f"[INFO] RetinaNet 모델 준비 완료. 총 클래스 수: {K}")
    
    # 6. 학습 루프 실행
    best_val_loss = float('inf')
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, DEVICE, K)
        val_loss = validate_loss(model, val_loader, DEVICE, K)
        lr_sch.step()
        
        print(f"[Epoch {epoch:02d}/{EPOCHS}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CKPT_DIR / "best_model.pth")
            print(f"  -> Best model saved with val_loss: {best_val_loss:.4f}")
            
    print(f"\n[INFO] 학습 완료. Best Val Loss: {best_val_loss:.4f}")
    
    # 7. 최종 단계 (평가 및 시각화 등)
    # (평가/시각화가 필요할 경우 아래 주석을 풀고 함수를 완성하세요)
    # print("\n[INFO] 최종 평가 시작...")
    # run_evaluation(model, val_ids, DEVICE, id2path, ann_by_img, COCO)
    
    # print("\n[INFO] 최종 시각화 시작...")
    # cat_id2name = {c["id"]: c["name"] for c in COCO["categories"]}
    # run_visualization(model, val_ids[:8], DEVICE, id2path, cat_id2name, CKPT_DIR)


if __name__ == '__main__':
    # 이 스크립트가 직접 실행될 때만 아래 코드가 동작합니다.
    
    # 실행 시 받을 인자(argument)들을 정의합니다.
    parser = argparse.ArgumentParser(description="RetinaNet Training Script")
    parser.add_argument('--data_root', type=str, required=True, help='데이터셋의 최상위 경로 (e.g., D:/datasets/pills)')
    parser.add_argument('--epochs', type=int, default=50, help='총 학습 에폭 수')
    parser.add_argument('--lr', type=float, default=0.005, help='학습률')
    parser.add_argument('--batch_size', type=int, default=4, help='배치 사이즈')
    
    # 정의된 인자들을 파싱합니다.
    args = parser.parse_args()
    
    # 파싱된 인자들을 main 함수에 전달하여 실행합니다.
    main(args)