import os
import json
import yaml

import pandas as pd
from PIL import Image

def save_csv():
    pred_dir = "v12_runs/predict/labels"
    img_dir = "../data/test_images"
    out_csv = "../submissions/submission.csv"

    rows = []
    annotation_id = 1

    # 🔑 매핑 불러오기
    class_map = load_class_map()

    for label_file in os.listdir(pred_dir):
        if not label_file.endswith(".txt"):
            continue

        image_id = os.path.splitext(label_file)[0]   # "1"
        img_path = os.path.join(img_dir, image_id + ".png")
        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir, image_id + ".jpg")

        with Image.open(img_path) as im:
            W, H = im.size

        with open(os.path.join(pred_dir, label_file)) as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 6:
                    cls, x, y, w, h, conf = map(float, parts)
                else:
                    continue  # conf 없는 라인 무시

                x_min = (x - w/2) * W
                y_min = (y - h/2) * H
                box_w = w * W
                box_h = h * H

                rows.append([
                    annotation_id,
                    int(image_id),
                    class_map[int(cls)],   # 🔑 여기서 변환
                    int(x_min),
                    int(y_min),
                    int(box_w),
                    int(box_h),
                    round(conf, 2)
                ])
                annotation_id += 1

    df = pd.DataFrame(rows, columns=[
        "annotation_id", "image_id", "category_id",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"
    ])
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


def load_class_map():
    anno_dir = "../data/train_annotations"
    name2id = {}

    # 하위 모든 폴더 순회
    for root, dirs, files in os.walk(anno_dir):
        for file in files:
            if not file.endswith(".json"):
                continue
            with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                data = json.load(f)
                if "categories" in data:
                    for cat in data["categories"]:
                        name2id[cat["name"].strip()] = cat["id"]

    # yolo_data.yaml 불러오기
    with open("../configs/yolo_data.yaml", "r", encoding="utf-8") as f:
        yolo_cfg = yaml.safe_load(f)
    names = [n.strip() for n in yolo_cfg["names"]]

    # YOLO 인덱스 → 실제 category_id 매핑
    class_map = {}
    for i, name in enumerate(names):
        if name not in name2id:
            raise KeyError(f"⚠️ '{name}' 가 JSON categories에 없음")
        class_map[i] = name2id[name]

    return class_map