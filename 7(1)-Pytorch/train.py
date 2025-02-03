import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Any
from resnet import ResNet, BasicBlock
from config import *
import torchvision

NUM_CLASSES = 10  

# Data augmentation 추가
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Dataset 로드 시 transform 적용
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                      download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                     download=True, transform=transform_test)

train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

# device 설정 변경
device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f"Using device: {device}")

# resnet 18 선언하기
## TODO
model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=NUM_CLASSES).to(device)

criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
optimizer: optim.Adam = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 학습 
def train(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> None:
    model.train()
    total_loss: float = 0
    correct: int = 0
    total: int = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy: float = 100. * correct / total
    print(f"Train Loss: {total_loss / len(loader):.4f}, Accuracy: {accuracy:.2f}%")

# 평가 
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> None:
    model.eval()
    total_loss: float = 0
    correct: int = 0
    total: int = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy: float = 100. * correct / total
    print(f"Test Loss: {total_loss / len(loader):.4f}, Accuracy: {accuracy:.2f}%")

# 학습 및 평가 루프
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    train(model, train_loader, criterion, optimizer, device)
    evaluate(model, test_loader, criterion, device)

# 모델 저장
torch.save(model.state_dict(), "resnet18_checkpoint.pth")
print(f"Model saved to resnet18_checkpoint.pth")
