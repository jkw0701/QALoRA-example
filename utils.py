"""
공통 유틸리티 함수
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from config import *

def get_mnist_loaders():
    """MNIST 데이터 로더 반환"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(DATA_DIR, train=True, 
                                  download=True, transform=transform)
    test_dataset = datasets.MNIST(DATA_DIR, train=False, 
                                 transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                            shuffle=False, num_workers=2)
    
    return train_loader, test_loader

def evaluate_model(model, test_loader, device=DEVICE):
    """모델 평가"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            data = data.view(data.size(0), -1)
            
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def save_model(model, path):
    """모델 저장"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'class_name': model.__class__.__name__,
        }
    }, path)
    print(f"Model saved to {path}")

def load_model(model_class, path, **kwargs):
    """모델 로드"""
    checkpoint = torch.load(path, map_location=DEVICE)
    model = model_class(**kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    print(f"Model loaded from {path}")
    return model

def plot_comparison(original, quantized, title="Comparison"):
    """원본과 양자화 비교 플롯"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 원본
    axes[0].hist(original.flatten().cpu().numpy(), bins=50, alpha=0.7)
    axes[0].set_title("Original")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Frequency")
    
    # 양자화
    axes[1].hist(quantized.flatten().cpu().numpy(), bins=50, alpha=0.7, color='orange')
    axes[1].set_title("Quantized")
    axes[1].set_xlabel("Value")
    
    # 오차
    error = (original - quantized).abs()
    axes[2].hist(error.flatten().cpu().numpy(), bins=50, alpha=0.7, color='red')
    axes[2].set_title("Absolute Error")
    axes[2].set_xlabel("Error")
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()