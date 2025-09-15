"""
Step 2: 기본 양자화 구현
Post-training quantization (PTQ) 적용
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from utils import get_mnist_loaders, evaluate_model, save_model, load_model, plot_comparison
from step1_baseline_model import BaselineModel

class QuantizationUtils:
    """양자화 유틸리티 함수들"""
    
    @staticmethod
    def compute_scale_zp(tensor, n_bits=8):
        """스케일과 영점 계산"""
        qmin = -(2**(n_bits-1))
        qmax = 2**(n_bits-1) - 1
        
        min_val = tensor.min()
        max_val = tensor.max()
        
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - torch.round(min_val / scale)
        
        return scale, zero_point
    
    @staticmethod
    def quantize(tensor, scale, zero_point, n_bits=8):
        """텐서 양자화"""
        qmin = -(2**(n_bits-1))
        qmax = 2**(n_bits-1) - 1
        
        q_tensor = torch.round(tensor / scale + zero_point)
        q_tensor = torch.clamp(q_tensor, qmin, qmax)
        
        return q_tensor
    
    @staticmethod
    def dequantize(q_tensor, scale, zero_point):
        """텐서 역양자화"""
        return (q_tensor - zero_point) * scale

class QuantizedLinear(nn.Module):
    """양자화된 Linear 레이어"""
    
    def __init__(self, original_layer, n_bits=8):
        super().__init__()
        self.n_bits = n_bits
        
        # 원본 가중치 양자화
        weight = original_layer.weight.data
        self.weight_scale, self.weight_zp = QuantizationUtils.compute_scale_zp(weight, n_bits)
        self.quantized_weight = QuantizationUtils.quantize(
            weight, self.weight_scale, self.weight_zp, n_bits
        )
        
        # 바이어스 처리
        if original_layer.bias is not None:
            self.bias = original_layer.bias.data.clone()
        else:
            self.bias = None
            
    def forward(self, x):
        # 가중치 역양자화
        weight = QuantizationUtils.dequantize(
            self.quantized_weight, self.weight_scale, self.weight_zp
        )
        
        # Linear 연산
        output = F.linear(x, weight, self.bias)
        return output

def quantize_model(model, n_bits=8):
    """모델 전체 양자화"""
    quantized_model = BaselineModel()
    
    # 각 레이어 양자화
    quantized_model.fc1 = QuantizedLinear(model.fc1, n_bits)
    quantized_model.fc2 = QuantizedLinear(model.fc2, n_bits)
    quantized_model.fc3 = QuantizedLinear(model.fc3, n_bits)
    
    return quantized_model

def main():
    print("="*50)
    print("Step 2: Basic Quantization")
    print("="*50)
    
    # Baseline 모델 로드
    model = load_model(BaselineModel, BASELINE_MODEL_PATH)
    
    # 테스트 데이터 로드
    _, test_loader = get_mnist_loaders()
    
    # 원본 정확도
    original_accuracy = evaluate_model(model, test_loader)
    print(f"Original Model Accuracy: {original_accuracy:.2f}%")
    
    # 다양한 비트로 양자화 테스트
    for n_bits in [8, 4, 2]:
        print(f"\n--- {n_bits}-bit Quantization ---")
        
        quantized_model = quantize_model(model, n_bits)
        quantized_model.to(DEVICE)
        
        # 정확도 평가
        quantized_accuracy = evaluate_model(quantized_model, test_loader)
        print(f"Quantized Model Accuracy: {quantized_accuracy:.2f}%")
        print(f"Accuracy Drop: {original_accuracy - quantized_accuracy:.2f}%")
        
        # 모델 크기 비교
        original_size = sum(p.numel() for p in model.parameters()) * 32 / 8 / (1024 * 1024)
        quantized_size = sum(p.numel() for p in model.parameters()) * n_bits / 8 / (1024 * 1024)
        print(f"Size Reduction: {original_size:.2f}MB → {quantized_size:.2f}MB "
              f"({original_size/quantized_size:.1f}x)")
        
        # 가중치 분포 시각화 (4비트일 때만)
        if n_bits == 4:
            original_weight = model.fc1.weight.data
            quantized_weight = QuantizationUtils.dequantize(
                quantized_model.fc1.quantized_weight,
                quantized_model.fc1.weight_scale,
                quantized_model.fc1.weight_zp
            )
            plot_comparison(original_weight, quantized_weight, 
                          f"{n_bits}-bit Weight Quantization")
            
            # 4비트 모델 저장
            save_model(quantized_model, QUANTIZED_MODEL_PATH)

if __name__ == "__main__":
    main()
    print("\nStep 2 completed successfully!")