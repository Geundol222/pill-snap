# 1️⃣ 라이브러리 및 설정

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
import matplotlib.pyplot as plt
import numpy as np
# ...existing code...
from PIL import Image
# ...existing code...

class CustomDataset(Dataset):
    def __init__(self, img_dir, annotations, transform=None):
        self.img_dir = img_dir
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name, target = self.annotations[idx]
        image = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")
        if self.transform:
            image = self.transform(image)
        boxes = torch.tensor(target['boxes'], dtype=torch.float32)
        labels = torch.tensor(target['labels'], dtype=torch.int64)
        # 데이터 유효성 체크
        if torch.isnan(boxes).any() or torch.isinf(boxes).any():
            print(f"[경고] nan/inf 박스 발견! idx={idx}, 파일={img_name}, boxes={boxes}")
        if (boxes < 0).any():
            print(f"[경고] 음수 박스 좌표 발견! idx={idx}, 파일={img_name}, boxes={boxes}")
        if boxes.numel() == 0 or labels.numel() == 0:
            print(f"[경고] 빈 박스/라벨 샘플! idx={idx}, 파일={img_name}")
        target_dict = {
            'boxes': boxes,
            'labels': labels
        }
        return image, target_dict


# 2️⃣ 실제 데이터 경로 지정 (사용자 지정 경로)
data_root = r'C:/Users/user/.vscode/pill-snap/data/ai04-level1-project'
train_images_dir = os.path.join(data_root, 'train_images')
test_images_dir = os.path.join(data_root, 'test_images')
train_annotations_dir = os.path.join(data_root, 'train_annotations')

# train_images, test_images, train_annotations 폴더 내 파일 개수 출력
train_img_files = [f for f in os.listdir(train_images_dir) if f.lower().endswith('.png')]
test_img_files = [f for f in os.listdir(test_images_dir) if f.lower().endswith('.png')]

ann_count = 0
for root, dirs, files in os.walk(train_annotations_dir):
    ann_count += len([f for f in files if f.lower().endswith('.json')])
print(f"train_images: {len(train_img_files)}개, test_images: {len(test_img_files)}개, train_annotations(json): {ann_count}개")


# 3️⃣ Custom Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((300,300))  # SSD 입력 크기 맞춤
])

class CustomDataset(Dataset):
    def __init__(self, img_dir, annotations, transform=None):
        self.img_dir = img_dir
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name, target = self.annotations[idx]
        image = plt.imread(os.path.join(self.img_dir, img_name))
        if self.transform:
            image = self.transform(image)
        target_dict = {
            'boxes': torch.tensor(target['boxes'], dtype=torch.float32),
            'labels': torch.tensor(target['labels'], dtype=torch.int64)
        }
        return image, target_dict

import json
# 실제 데이터셋 생성 예시
# 어노테이션 json 파일을 읽어서 (이미지파일명, 어노테이션) 튜플 리스트 생성
annotations = []
import fnmatch
# 1. 전체 category_id 수집
category_ids = set()
for root, dirs, files in os.walk(train_annotations_dir):
    for ann_file in fnmatch.filter(files, '*.json'):
        ann_path = os.path.join(root, ann_file)
        try:
            with open(ann_path, 'r', encoding='utf-8') as f:
                ann_data = json.load(f)
            for ann in ann_data.get('annotations', []):
                category_ids.add(ann['category_id'])
        except Exception as e:
            print(f"[어노테이션 파싱 오류] 파일: {ann_path} | 에러: {e}")

# 2. category_id → label 인덱스 매핑 생성
category_ids = sorted(list(category_ids))
catid2label = {catid: idx for idx, catid in enumerate(category_ids)}
print(f"category_id to label mapping: {catid2label}")

# 3. 실제 어노테이션 파싱
for root, dirs, files in os.walk(train_annotations_dir):
    for ann_file in fnmatch.filter(files, '*.json'):
        ann_path = os.path.join(root, ann_file)
        try:
            with open(ann_path, 'r', encoding='utf-8') as f:
                ann_data = json.load(f)
            if 'images' in ann_data and len(ann_data['images']) > 0:
                img_name = ann_data['images'][0]['file_name']
            else:
                img_name = ann_file.replace('.json', '.png')
            raw_boxes = [ann['bbox'] for ann in ann_data.get('annotations', [])]
            raw_labels = [ann['category_id'] for ann in ann_data.get('annotations', [])]
            boxes, labels = [], []
            for box, label in zip(raw_boxes, raw_labels):
                # box: [xmin, ymin, xmax, ymax] 또는 [x, y, w, h] 형식일 수 있음
                if len(box) == 4:
                    x1, y1, x2, y2 = box
                    # 만약 COCO 형식([x, y, w, h])이면 변환
                    if x2 < x1 or y2 < y1:
                        # swap 또는 변환
                        x1, x2 = min(x1, x2), max(x1, x2)
                        y1, y2 = min(y1, y2), max(y1, y2)
                    width = x2 - x1
                    height = y2 - y1
                    if width > 0 and height > 0:
                        # category_id를 label 인덱스로 변환
                        mapped_label = catid2label.get(label, None)
                        if mapped_label is not None:
                            boxes.append([x1, y1, x2, y2])
                            labels.append(mapped_label)
                        else:
                            print(f"[경고] 알 수 없는 category_id {label} → 무시됨. 파일: {ann_path}")
                    else:
                        print(f"[경고] 잘못된 bbox {box} (width={width}, height={height}) → 무시됨. 파일: {ann_path}")
            if len(boxes) > 0:
                annotations.append((img_name, {'boxes': boxes, 'labels': labels}))
        except Exception as e:
            print(f"[어노테이션 파싱 오류] 파일: {ann_path} | 에러: {e}")

# 어노테이션 개수와 예시 출력
print(f"어노테이션 개수: {len(annotations)}")
if len(annotations) > 0:
    print("첫 번째 어노테이션 예시:", annotations[0])
if len(annotations) == 0:
    raise ValueError("어노테이션이 0개입니다. 경로, 폴더 구조, json 파싱 코드를 확인하세요.")

# CustomDataset 인스턴스 생성 (dataset 변수 정의)
dataset = CustomDataset(train_images_dir, annotations, transform=transform)
# 4️⃣ collate_fn 정의
def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)



# 5️⃣ 학습 루프
def train_model(dataset, epochs=1, batch_size=2, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 실제 클래스 개수 자동 지정
    num_classes = len(catid2label)
    print(f"[INFO] 모델 num_classes: {num_classes}")
    model = ssd300_vgg16(weights=None, num_classes=num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    model.train()
    for epoch in range(epochs):
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {losses.item():.4f}")
    return model


# 6️⃣ simple mAP 계산 (IoU 기반)
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB-xA) * max(0, yB-yA)
    boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    iou = interArea / (boxAArea + boxBArea - interArea + 1e-6)
    return iou

def simple_map(model, dataset, iou_threshold=0.5):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    device = next(model.parameters()).device
    model.eval()
    
    total_tp, total_fp, total_fn = 0, 0, 0
    
    with torch.no_grad():
        for idx, (images, targets) in enumerate(loader):
            images = [img.to(device) for img in images]
            outputs = model(images)
            
            pred_boxes = outputs[0]['boxes'].cpu()
            pred_labels = outputs[0]['labels'].cpu()
            true_boxes = targets[0]['boxes']
            true_labels = targets[0]['labels']

            # 예측/정답 샘플 일부 출력 (최초 3개만)
            if idx < 3:
                print(f"\n[샘플 {idx}] 예측 박스/라벨 개수: {len(pred_boxes)}, 정답 박스/라벨 개수: {len(true_boxes)}")
                print(f"예측 라벨: {pred_labels.tolist()}")
                print(f"정답 라벨: {true_labels.tolist()}")
                if len(pred_boxes) > 0:
                    print(f"예측 박스(첫 1개): {pred_boxes[0].tolist()}")
                if len(true_boxes) > 0:
                    print(f"정답 박스(첫 1개): {true_boxes[0].tolist()}")

            matched = []
            tp = 0
            for i, pb in enumerate(pred_boxes):
                for j, tb in enumerate(true_boxes):
                    if j in matched:
                        continue
                    if pred_labels[i] != true_labels[j]:
                        continue
                    if compute_iou(pb, tb) >= iou_threshold:
                        tp += 1
                        matched.append(j)
                        break
            fp = len(pred_boxes) - tp
            fn = len(true_boxes) - tp
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
    
    ap = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    print(f"Simple mAP (IoU>={iou_threshold}): {ap:.4f}, Recall: {recall:.4f}")


# 7️⃣ 테스트 루프
def test_loop(model, dataset, save_dir="./results"):
    os.makedirs(save_dir, exist_ok=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    device = next(model.parameters()).device
    model.eval()
    
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            images = [img.to(device) for img in images]
            plt.imshow(images[0].cpu().permute(1,2,0))
            plt.axis('off')
            plt.savefig(os.path.join(save_dir, f"test_{i}.png"))
            plt.close()


# 8️⃣ 실행
if __name__=="__main__":
    model = train_model(dataset, epochs=50, batch_size=16)
    print("\n[모델 평가: simple mAP]")
    simple_map(model, dataset)
    test_loop(model, dataset)