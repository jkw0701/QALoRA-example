"""
Step 5: TALSQ (Temporal Activation LSQ) 구현
시간/배치별 활성화 스케일 학습
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import *
from utils import get_mnist_loaders, evaluate_model, save_model, load_model
from step3_qalora_layer import STEFunction
from step1_baseline_model import BaselineModel

class TALSQLinear(nn.Module):
    """TALSQ를 적용한 Linear 레이어"""
    
    def __init__(self, in_features, out_features, rank=8, n_bits=4, n_steps=10):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.n_bits = n_bits
        self.n_steps = n_steps
        
        # Pretrained weights
        self.register_buffer('pretrained_weight', 
                           torch.randn(out_features, in_features))
        
        # LoRA parameters
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Weight quantization scales
        self.weight_scale = nn.Parameter(torch.ones(out_features) * 0.1)
        
        # TALSQ: Temporal activation scales
        self.temporal_scales = nn.Parameter(torch.ones(n_steps) * 0.1)
        
        # Bias
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        self.current_step = 0
        
    def set_step(self, step):
        """Set current temporal step"""
        self.current_step = min(step, self.n_steps - 1)
        
    def forward(self, x):
        # Merge LoRA with pretrained weights
        lora_weight = self.lora_B @ self.lora_A
        merged_weight = self.pretrained_weight + lora_weight
        
        # Quantize weights
        quantized_weights = []
        for i in range(self.out_features):
            w_channel = merged_weight[i:i+1, :]
            w_quant = STEFunction.apply(
                w_channel,
                self.weight_scale[i],
                torch.zeros(1).to(w_channel.device),
                self.n_bits
            )
            quantized_weights.append(w_quant)
        
        weight_quantized = torch.cat(quantized_weights, dim=0)
        
        # TALSQ: Use temporal scale for activations
        activation_scale = self.temporal_scales[self.current_step]
        x_quantized = STEFunction.apply(
            x,
            activation_scale,
            torch.zeros(1).to(x.device),
            self.n_bits
        )
        
        output = F.linear(x_quantized, weight_quantized, self.bias)
        return output

class TALSQModel(nn.Module):
    """Model with TALSQ layers"""
    
    def __init__(self, pretrained_model=None, rank=8, n_bits=4, n_steps=10):
        super().__init__()
        self.n_steps = n_steps
        
        self.fc1 = TALSQLinear(INPUT_DIM, HIDDEN_DIM, rank, n_bits, n_steps)
        self.fc2 = TALSQLinear(HIDDEN_DIM, HIDDEN_DIM, rank, n_bits, n_steps)
        self.fc3 = TALSQLinear(HIDDEN_DIM, OUTPUT_DIM, rank, n_bits, n_steps)
        self.dropout = nn.Dropout(0.2)
        
        if pretrained_model is not None:
            self.initialize_from_pretrained(pretrained_model)
    
    def initialize_from_pretrained(self, model):
        """Initialize from pretrained model"""
        self.fc1.pretrained_weight.copy_(model.fc1.weight.data)
        self.fc2.pretrained_weight.copy_(model.fc2.weight.data)
        self.fc3.pretrained_weight.copy_(model.fc3.weight.data)
        
        self.fc1.bias.data.copy_(model.fc1.bias.data)
        self.fc2.bias.data.copy_(model.fc2.bias.data)
        self.fc3.bias.data.copy_(model.fc3.bias.data)
    
    def set_step(self, step):
        """Set temporal step for all layers"""
        self.fc1.set_step(step)
        self.fc2.set_step(step)
        self.fc3.set_step(step)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def train_talsq():
    print("="*50)
    print("Step 5: TALSQ Implementation")
    print("="*50)
    
    # Load pretrained model
    pretrained_model = load_model(BaselineModel, BASELINE_MODEL_PATH)
    
    # Create TALSQ model
    n_steps = 10  # Simulate temporal steps
    model = TALSQModel(pretrained_model, rank=LORA_RANK, n_bits=N_BITS, n_steps=n_steps)
    model.to(DEVICE)
    
    # Data loaders
    train_loader, test_loader = get_mnist_loaders()
    
    # Optimizer
    params = [p for n, p in model.named_parameters() 
             if 'lora' in n or 'scale' in n or 'temporal' in n]
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Number of temporal steps: {n_steps}")
    print(f"Trainable parameters: {sum(p.numel() for p in params):,}")
    
    # Training with temporal steps
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(DEVICE).view(-1, INPUT_DIM)
            target = target.to(DEVICE)
            
            # Simulate different temporal steps
            step = batch_idx % n_steps
            model.set_step(step)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 99:
                print(f'Epoch [{epoch+1}/{EPOCHS}], '
                      f'Step [{batch_idx+1}/{len(train_loader)}], '
                      f'Temporal Step [{step}], '
                      f'Loss: {running_loss/100:.4f}')
                running_loss = 0.0
        
        # Evaluation across different steps
        accuracies = []
        for step in range(n_steps):
            model.set_step(step)
            acc = evaluate_model(model, test_loader)
            accuracies.append(acc)
        
        avg_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        
        print(f'Epoch [{epoch+1}/{EPOCHS}]')
        print(f'  Average Accuracy: {avg_accuracy:.2f}% (±{std_accuracy:.2f}%)')
        print(f'  Min/Max: {min(accuracies):.2f}% / {max(accuracies):.2f}%')
    
    # Save model
    save_model(model, TALSQ_MODEL_PATH)
    
    # Analyze temporal scales
    print("\n--- Temporal Scale Analysis ---")
    for name, module in model.named_modules():
        if hasattr(module, 'temporal_scales'):
            scales = module.temporal_scales.detach().cpu().numpy()
            print(f"{name}:")
            print(f"  Scales: {scales}")
            print(f"  Variation: {scales.std() / scales.mean() * 100:.2f}%")
    
    return model

if __name__ == "__main__":
    model = train_talsq()
    print("\nStep 5 completed successfully!")