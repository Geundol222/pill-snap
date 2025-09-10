import os


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, train_loader, val_loader, optimizer, epochs, lr_scheduler, device, iou_threshold=0.5, score_threshold=0.5):
    """
    train

    Arg:
        model: 학습할 모델
        train_loader: 학습 데이터 로더
        val_loader: 검증 데이터 로더
        optimizer: 최적화 함수
        epochs (int): 학습 epoch 수
        lr_scheduler: learning rate 스케줄러
        device: 실행 디바이스 (cuda/cpu)
        iou_threshold (float): IoU 기준값
        score_threshold (float): confidence 기준값

    Description:
        train 함수입니다.
        해당 함수에서 validation은 TP, FP, FN을 출력하도록 설계되었습니다.
        추후에 RetinaNet의 Focal Loss 튜닝을 위해서는 FP vs FN Trade off 를 확인하면 좋습니다.
    """
    for epoch in range(epochs):
        model.train()
        train_cost = 0.0

        for images, targets in tqdm(train_loader, desc='Training'):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            train_cost += losses.item()

            losses.backward()
            optimizer.step()

        lr_scheduler.step()
        avg_train_loss = train_cost / len(train_loader)
        print(f'Epoch: {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}')

        # -------------------- Validation --------------------
        model.eval()
        TP, FP, FN = 0, 0, 0

        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc='Validation'):
                images = [img.to(device) for img in images]
                preds = model(images)

                for pred, target in zip(preds, targets):
                    pred_boxes = pred['boxes'].cpu()
                    pred_scores = pred['scores'].cpu()
                    gt_boxes = target['boxes'].cpu()

                    matched = set()
                    for pb, score in zip(pred_boxes, pred_scores):
                        if score < score_threshold:  # confidence 필터
                            continue
                        ious = torchvision.ops.box_iou(pb.unsqueeze(0), gt_boxes)
                        max_iou, idx = ious.max(dim=1)
                        if max_iou >= iou_threshold and idx.item() not in matched:
                            TP += 1
                            matched.add(idx.item())
                        else:
                            FP += 1
                    FN += len(gt_boxes) - len(matched)

        precision = TP / (TP + FP + 1e-6)
        recall = TP / (TP + FN + 1e-6)
        print(f"Validation Precision: {precision:.4f}, Recall: {recall:.4f}")


def tuned_model(class_num=5):
    """
    모델 튜닝 함수

    Arg:
        class_num: 클래스 수

    Description:
        모델의 하이퍼파라미터 튜닝을 위해서 함수로 따로 분류하였습니다.
        만약 하이퍼파라미터의 수정이 필요한경우 해당 함수에서 조정 하면 됩니다.

        현재는 BaseLine을 잡기위해서 최대한 RetinaNet의 기본값으로 파라미터를 설정하였고,
        어떤 파라미터를 수정하면 좋은지에 대해서 명시하기 위해 파라미터를 설정하였습니다.
    """

    # RetinaNet의 기본 anchor_generator 필요시 튜닝가능
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )

    model = retinanet_resnet50_fpn_v2(
        weights=None,
        weights_backbone=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1,
        num_classes=5,
        anchor_generator=anchor_generator
    )

    model.head.classification_head.num_classes = class_num

    # model의 Focal Loss 튜닝
    model.head.classification_head.focal_loss.gamma = 2.0
    model.head.classification_head.focal_loss.alpha = 0.25

    return model


def main():
    """
    Main 실행함수

    Description:
        모델을 실제로 돌리는 함수입니다.
        EDA와 전처리의 return 값에 따라 자유롭게 변경하면됩니다.

        optimizer와 lr_scheduler도 하이퍼파라미터의 하나이므로 변경이 가능합니다.
    """

    model = tuned_model(class_num=5)
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=0.0001)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    print("hello")

if __name__ == "__main__":
    main()