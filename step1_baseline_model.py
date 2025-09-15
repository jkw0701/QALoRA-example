"""
Step 1: Baseline 모델 학습
기본 FP32 모델을 학습하여 비교 기준 설정
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import *
from utils import get_mnist_loaders, evaluate_model, save_model

class BaselineModel(nn.Module):
    """기본 신경망 모델"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc3 = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def train_baseline():
    """Baseline 모델 학습"""
    print("="*50)
    print("Step 1: Training Baseline Model")
    print("="*50)
    
    # 데이터 로드
    train_loader, test_loader = get_mnist_loaders()
    
    # 모델 초기화
    model = BaselineModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # 학습
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(DEVICE).view(-1, INPUT_DIM)
            target = target.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 99:
                print(f'Epoch [{epoch+1}/{EPOCHS}], '
                      f'Step [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {running_loss/100:.4f}')
                running_loss = 0.0
        
        # 평가
        accuracy = evaluate_model(model, test_loader)
        print(f'Epoch [{epoch+1}/{EPOCHS}] Test Accuracy: {accuracy:.2f}%')
    
    # 모델 저장
    save_model(model, BASELINE_MODEL_PATH)
    
    # 모델 크기 출력
    total_params = sum(p.numel() for p in model.parameters())
    model_size = total_params * 4 / (1024 * 1024)  # FP32 기준
    print(f"\nBaseline Model Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: {model_size:.2f} MB")
    print(f"Final Test Accuracy: {accuracy:.2f}%")
    
    return model

if __name__ == "__main__":
    model = train_baseline()
    print("\nStep 1 completed successfully!")