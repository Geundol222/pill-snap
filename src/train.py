import random
import os
from itertools import product

import pandas as pd
from ultralytics import YOLO

def train(model_size: str='m', epochs: int=50, random: bool=False, num_trials: int=5):
    """
    Main train 함수

    Args:
        - model_size: YOLO모델의 사이즈 [n, s, m] => l과 xl은 크기가 너무 커서 제외
        - epochs: 실행할 에폭 수
        - random: Random Search 여부
        - num_trials: Random Search 사용시, 몇개의 조합을 테스트할지 결정하는 변수

    Description:
        Random Search를 위한 함수와 일반 모델링 함수를 나눴습니다.
    """



    model = YOLO(f'yolo12{model_size}.pt')

    if random:
        random_search(model, epochs, num_trials)
    else:
        model_train(model, epochs)

def model_train(model, epochs=50):
    """
    Model 실행함수

    Description:
        Yolo의 경우 augmentation이나 여러 전처리가 전부 라이브러리 안에 내장되어있고 학습시 알아서 전처리를 해주기때문에 편합니다.
        또한 train함수를 제작할 필요 없이 model.train으로 파라미터만 잘 설정해 주면 되겠습니다.

        더 높은 학습 확률을 위해 파라미터에서 augmentation 옵션을 바꿀 수 있습니다.

        - 예시
        model.train(
            data='../../configs/yolo_data.yaml',
            epochs=100,         # 에폭 수
            imgsz=640,          # 이미지 resize(정사각형)
            batch=16,           # 배치 사이즈(DataLoader의 그것)
            degrees=10,         # RandomRoation
            scale=0.5,          # RandomScale
            fliplr=0.5,         # RandomHorizontalFlip
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,          # hsv 3종: ColorJitter와 비슷
            mosaic=1.0,         # 4장의 이미지를 랜덤하게 합쳐서 한장으로 만듬
            mixup=0.2,          # 두장의 이미지를 섞어서 새로운 이미지를 만듬 -> 작은 데이터 셋에서 과적합 방지
            project="v12_runs", # yolo model은 학습을 돌릴경우 디렉토리 에 runs폴더를 만들게 되는데, 버전마다 다르게 생성하기 위해 이름을 커스텀하여 폴더를 생성할 수 있습니다.
            name="v12_train",   # 해당 작업이 어떤 작업인지를 명시해서 폴더링할 수도 있습니다. 이 기능은 사용자의 선택이지만, 기본적으로 train, val, predict의 폴더가 작업마다 생성되므로 꼭 필요한 파라미터는 아닙니다.
            exist_ok=True       # 이미 폴더가 있어도 덮어쓰기 허용
        )
    """
    
    close_mosaic_ratio = int(epochs * 0.3)

    model.train(
        data='./configs/yolo_data.yaml',
        epochs=epochs,
        close_mosaic=close_mosaic_ratio,
        imgsz=960,  # 640
        batch=4,    # 16
        nbs=16,     # nbs: 배치는 4인데 16의 효과를 주기위해 epoch마다 backward하는게 아니라 4번마다 한번 한다
        auto_augment='autoaugment', # 'randaugment'는 기본값 'autoaugment'또는 'augmix'
        cache=False,
        device=0,
        project="v12_runs",
        seed=42,
        exist_ok=True
    )

def random_search(model, epochs=50, num_trials=5):
    """
    Random_Search 함수

    Description:        
        이 함수에서 현재 서치를 통해 변경하고 있는 augmentation은 다음과 같습니다
        - mixup_options = [0.0, 0.1, 0.2]
        - mosaic_options = [0.0, 0.5, 1.0]
        - cutmix_options = [0.0, 0.1, 0.3]
        - copy_paste_options = [0.0, 0.1, 0.2]
        - augment_options = ['autoaugment', 'augmix', 'randaugment']
        - nbs_options = [8, 16, 32]
        - lr0_options = [0.001, 0.003, 0.005]

        num_trials는 몇개의 조합을 실험해 볼 지에 대한 변수입니다.

        Random Search는 조합 실험용이기 때문에 이미지사이즈를 960에서 768로 줄였습니다.

    """

    close_mosaic_ratio = int(epochs * 0.3)

    mixup_options = [0.0, 0.1, 0.2]
    mosaic_options = [0.0, 0.5, 1.0]
    cutmix_options = [0.0, 0.1, 0.3]
    copy_paste_options = [0.0, 0.1, 0.2]
    augment_options = ['autoaugment', 'augmix', 'randaugment']
    nbs_options = [8, 16, 32]
    lr0_options = [0.001, 0.003, 0.005]

    search_space = list(product(mixup_options, mosaic_options, cutmix_options, copy_paste_options, augment_options, nbs_options, lr0_options))
    num_trials = min(num_trials, len(search_space))
    sampled_trials = random.sample(search_space, num_trials)

    trial_results = []

    for i, (mixup, mosaic, cutmix, copy_paste, augment, nbs, lr0) in enumerate(sampled_trials, 1):
        trial_name = f"trial_{i}"
        print(f'Random Trial {i}/{num_trials}: '
            f'mixup={mixup}, mosaic={mosaic}, cutmix={cutmix}, copy_paste={copy_paste}, augment={augment}, nbs={nbs}, lr0={lr0}')
        
        model = YOLO('yolo12s.pt')

        model.train(
            data='./configs/yolo_data.yaml',
            epochs=epochs,
            close_mosaic=close_mosaic_ratio,
            imgsz=768,
            batch=4,
            nbs=nbs,
            auto_augment=augment,
            mixup=mixup,
            mosaic=mosaic,
            cutmix=cutmix,
            copy_paste=copy_paste,
            lr0=lr0,
            device=0,
            workers=0,
            project='v12_runs_random',
            name=trial_name,
            seed=42,
            exist_ok=True
        )

        result_file = f'v12_runs_random/{trial_name}/results.csv'
        if os.path.exists(result_file):        
            df = pd.read_csv(result_file)
            if "metrics/mAP50-95(B)" in df.columns:
                final_map = df["metrics/mAP50-95(B)"].iloc[-1]
            else:
                final_map = df["metrics/mAP50(B)"].iloc[-1]  # fallback

            # trial_name, mAP, 조합을 저장
            trial_results.append((trial_name, final_map, {
                "mixup": mixup,
                "mosaic": mosaic,
                "cutmix": cutmix,
                "augment": augment,
                "nbs": nbs,
                "lr0": lr0
            }))

    if trial_results:
        best_trial = max(trial_results, key=lambda x: x[1])
        print(f"🏆 Best Trial: {best_trial[0]} with mAP50-95={best_trial[1]:.4f}")
        print("   ⚙️ Best Params:", best_trial[2])