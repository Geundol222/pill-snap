import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# 이 주석 지울것 여기는 getcifar을 이용한 ResNet18 학습 예제
"""
ResNet 구현
Class BasicBlock           # ResNet-18, 34용 블록
Class Bottleneck           # ResNet-50 이상용 블록
Class ResNet               # 전체 ResNet 모델 클래스
def resnet18~resnet152     # 모델 빌더 함수

임포트 예시:
from resnetSSD import resnet18, resnet50

model = resnet18(num_classes=10)  # CIFAR-10 예제
"""

"""
getcifar을 이용한 ResNet18 학습 예제이며 epoch까지 되는건 확인테스트
여기는 .py들을 나누지않고 하나에 다 넣은 하나의.py 형태
각 기능별로 나누어진 .py들은 각각 data.py, train.py, test.py, models/resnetSSD.py, main.py
"""
# ResNet-18, 34에서 사용하는 블록 (50층이하)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        in_channels : 입력 feature map 채널 수
        out_channels: 출력 feature map 채널 수
        stride      : 다운샘플링 여부 결정 (stride>1이면 공간 크기 축소)
        downsample  : 입력 x를 identity로 사용하기 위해 차원 맞추기용 layer
        """
        super(BasicBlock, self).__init__()

        # 첫 번째 3x3 convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) # 배치 정규화
        self.relu = nn.ReLU(inplace=True)       # ReLU 활성화 함수

        # 두 번째 3x3 convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample    # 차원 맞추기용 layer
        self.stride = stride

    def forward(self, x):
        identity = x    # 입력 x를 identity로 저장, 스킵 연결용 입력

        # 순차적으로 Convolution-BatchNorm-ReLU 적용
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        """
        다운샘플링이 필요한 경우 (채널 수 또는 공간 크기가 변경되는 경우)
        입력 x에 대해 1x1 컨볼루션과 배치 정규화를 적용하여 차원을 맞춤
        """
        if self.downsample is not None:
            identity = self.downsample(x)

        # identity mapping후 ReLU 활성화 함수 적용(출력값이 음수가 되지 않도록)
        out += identity
        # ReLU를 넣지않는다면 out += identity를 통과 후 이 블록은 선형에 가까워지기때문에 ReLU를 넣어 비선형을 유지
        out = self.relu(out)

        return out

#ResNet-50, 101, 152에서 사용하는 블록 (50층 이상)
class Bottleneck(nn.Module):
    expansion = 4 # 여러 실험 결과 4배가 제일 안정적이라고함 (ResNet 논문 참고)

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        # 1x1, 3x3, 1x1 컨볼루션으로 구성된 병목 구조
        # 1x1 conv: 채널 수 축소
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3 conv: 공간적 특징 추출
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 conv: 채널 수 복원확장
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        """
        block       : 사용할 블록 (BasicBlock 또는 Bottleneck)
        layers      : 스테이지별 블록 개수 리스트 (예: [2, 2, 2, 2] for ResNet-18)
        num_classes : 최종 분류할 클래스 수
        """
        super(ResNet, self).__init__()

        self.in_channels = 64 # 첫 번째 컨볼루션 레이어의 출력 채널 수

        # 초기 Convoluion Layer, BatchNorm, ReLU, MaxPooling 구성
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual Layer 구성
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 분류기 (AdaptiveAvgPool -> Dropout -> Fully Connected)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)  # 드롭아웃 레이어 추가 (p는 드롭아웃 확률) 오버피팅 방지
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 가중치 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # 각 스테이지 별 레이어 생성
    def _make_layer(self, block, out_channels, blocks, stride=1):
        """
        block       : BasicBlock/Bottleneck
        out_channels: 레이어 출력 채널
        blocks      : 레이어에 쌓을 블록 수
        stride      : 첫 번째 블록의 stride (다운샘플링)
        """
        downsample = None # None인 이유

        # stride가 1이 아니거나 입력/출력 채널 수가 다르면 다운샘플링 수행
        # 입력과 출력 크기/채널이 다르면 identity 차원 맞춤
        # Basic block -> 18, 34-layer의 stage 1에서만 None
        # BottleNeck -> 50, 101, 152-layer는 모든 stage에서 항상 여기로 들어옴
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        # 첫 번째 블록은 Stride와 다운샘플링 적용
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        # 나머지 블록들은 Stride=1, 다운샘플링 없음
        for _ in range(1, blocks): # _는 반복변수 사용 안할 때 관례적으로 사용 늘 사용하는 i를 써도되지만 경고 방지
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers) # *연산자 -> 해당 리스트를 언패킹하여 Sequential에 전달

    def forward(self, x):
        # 초기 Convolution Layer, BatchNorm, ReLU, MaxPooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual Layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 분류기 (Global Average Pooling -> Dropout -> Fully Connected)
        x = self.avgpool(x)
        x = self.dropout(x) # 오버피팅 방지용 드롭아웃 레이어 추가 (학습 시에만 적용됨)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
# ResNet 모델 생성 함수 (간편 호출)
def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def resnet34(num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def resnet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

def resnet101(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)

def resnet152(num_classes=1000):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)
# 여기까지가 resnetSSD.py


def get_cifar10_dataloaders(batch_size=128, num_workers=2):
    """
    데이터 전처리 
    CIFAR-10 데이터셋용 DataLoader 생성 함수

    batch_size: 학습 배치 크기 (기본값 128)
    num_workers: 데이터 로딩 시 사용하는 서브 프로세스 수 (기본값 2)
    
    train_loader: 학습용 DataLoader
    test_loader: 테스트용 DataLoader
    """
    # 학습 데이터 전처리
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),   # 32x32 이미지 랜덤 Crop, 패딩 4픽셀 추가 → 데이터 증강
        transforms.RandomHorizontalFlip(),      # 좌우 반전 → 데이터 다양성 증가
        transforms.ToTensor(),                  # PIL Image → Tensor 변환
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # 채널별 평균값, 표준편차
        ])
    
    # 테스트 데이터 전처리
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
    # CIFAR-10 데이터셋 다운로드 및 생성
    train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    test_dataset  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    
    # DataLoader 생성
    # 학습용 DataLoader, shuffle=True로 매 epoch마다 데이터 섞기
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # 테스트용 DataLoader, shuffle=False로 순서 유지
    test_loader  = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)
    
    return train_loader, test_loader
# 여기까지가 data.py


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    한 epoch 동안 학습 수행

    model: 학습할 모델
    train_loader: 학습용 DataLoader
    criterion: 손실 함수 (예: CrossEntropyLoss)
    optimizer: 최적화 함수 (예: SGD, Adam)
    device: 연산 디바이스 (CPU or GPU)
    epoch: 현재 epoch 번호 (출력용)
    """    
    model.train()
    running_loss, correct, total = 0.0, 0, 0    # 손실, 맞춘 개수, 전체 개수 초기화

    for inputs, targets in train_loader:
        # inputs과 targets을 device로 이동 (GPU 사용 시)
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()                           # 이전 gradient 초기화
        outputs = model(inputs)                         # 모델 순전파
        loss = criterion(outputs, targets)              # 손실 계산
        loss.backward()                                 # 역전파
        optimizer.step()                                # 가중치 업데이트

        running_loss += loss.item()                     # 배치별 손실 누적
        
        # 예측값과 정답 비교
        _, predicted = outputs.max(1)                   # 클래스별 최대 값 인덱스 추출
        total += targets.size(0)                        # 전체 샘플 개수 누적
        correct += predicted.eq(targets).sum().item()   # 맞춘 개수 누적

    # epoch 단위 정확도 계산
    acc = 100. * correct / total
    # 평균 손실과 정확도 출력
    print(f"Epoch [{epoch}] Train Loss: {running_loss/len(train_loader):.4f}, Acc: {acc:.2f}%")
# 여기까지가 train.py


def test_one_epoch(model, test_loader, criterion, device, epoch):
    """
    한 epoch 동안 테스트(검증) 수행

    model: 평가할 모델
    test_loader: 테스트용 DataLoader
    criterion: 손실 함수 (예: CrossEntropyLoss)
    device: 연산 디바이스 (CPU or GPU)
    epoch: 현재 epoch 번호 (출력용)
    """
    model.eval()
    running_loss, correct, total = 0.0, 0, 0   # 손실, 맞춘 개수, 전체 개수 초기화

    with torch.no_grad(): # 평가 시에는 gradient 계산 불필요 (메모리 절약)
        for inputs, targets in test_loader:
            # inputs과 targets을 device로 이동 (GPU 사용 시)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)                         # 모델 순전파
            loss = criterion(outputs, targets)              # 손실 계산

            running_loss += loss.item()                     # 배치별 손실 누적

            # 예측값과 정답 비교
            _, predicted = outputs.max(1)                   # 클래스별 최대 값 인덱스 추출
            total += targets.size(0)                        # 전체 샘플 개수 누적
            correct += predicted.eq(targets).sum().item()   # 맞춘 개수 누적

    # epoch 단위 정확도 계산
    acc = 100. * correct / total
    # 평균 손실과 정확도 출력
    print(f"Epoch [{epoch}] Test Loss: {running_loss/len(test_loader):.4f}, Acc: {acc:.2f}%")
# 여기까지가 test.py


if __name__ == "__main__":
    """
    메인 실행 파일

    CIFAR-10 데이터셋을 사용하여 ResNet-18 모델 학습 및 평가
    학습 및 테스트 함수는 각각 train.py와 test.py에서 임포트
    데이터 로딩은 data.py의 get_cifar10_dataloaders 함수 사용
    모델은 models/resnetSSD.py의 resnet18 함수로 생성
    """

    # 모델 선택 (예: ResNet-18)
    # GPU가 사용 가능하면 'cuda', 없으면 'cpu'를 사용
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터
    # get_cifar10_dataloaders 함수를 호출하여 CIFAR-10 데이터셋 로딩
    # batch_size=128로 학습 데이터를 한 번에 128개씩 가져옴
    train_loader, test_loader = get_cifar10_dataloaders(batch_size=128)

    # 모델
    # ResNet-18 모델 생성, CIFAR-10은 클래스 수가 10
    # to(device)를 통해 GPU 또는 CPU에 모델을 올림
    model = resnet18(num_classes=10).to(device)

    # 손실함수 & 옵티마이저
    # CrossEntropyLoss: 분류 문제에 적합한 손실 함수
    criterion = nn.CrossEntropyLoss()
    # SGD: 확률적 경사 하강법, momentum=0.9, 가중치 감쇠(weight decay)=5e-4
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    # StepLR: 일정 epoch마다 learning rate를 gamma만큼 감소
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # 학습 루프
    num_epochs = 50 # 총 학습 epoch 수 50
    for epoch in range(1, num_epochs+1):
        train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)   # 한 epoch 동안 학습 수행
        test_one_epoch(model, test_loader, criterion, device, epoch)               # 한 epoch 동안 테스트 수행
        scheduler.step()        # 스케줄러로 learning rate 조정
# 여기까지가 main.py