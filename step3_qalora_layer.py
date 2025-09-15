"""
Step 3: QALoRA 레이어 구현
LoRA와 양자화를 결합한 레이어 - 개선된 버전
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import *
from utils import get_mnist_loaders, evaluate_model, save_model, load_model
from step1_baseline_model import BaselineModel

class STEFunction(torch.autograd.Function):
    """Straight-Through Estimator for gradient"""
    @staticmethod
    def forward(ctx, input, scale, zero_point, n_bits):
        ctx.save_for_backward(input, scale)
        ctx.n_bits = n_bits
        ctx.zero_point = zero_point
        
        qmin = -(2**(n_bits-1))
        qmax = 2**(n_bits-1) - 1
        
        # Quantize
        output = torch.round(input / scale + zero_point)
        output = torch.clamp(output, qmin, qmax)
        
        # Dequantize  
        output = (output - zero_point) * scale
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, scale = ctx.saved_tensors
        n_bits = ctx.n_bits
        zero_point = ctx.zero_point
        
        # Gradient for input - with clipping awareness
        qmin = -(2**(n_bits-1))
        qmax = 2**(n_bits-1) - 1
        
        # Check if values are within quantization range
        q_input = input / scale + zero_point
        mask = (q_input >= qmin) & (q_input <= qmax)
        grad_input = grad_output * mask.float()
        
        # Gradient for scale
        grad_scale = None  # Will be computed by autograd
        
        return grad_input, grad_scale, None, None

class QALoRALinear(nn.Module):
    """Improved QALoRA Linear layer"""
    
    def __init__(self, in_features, out_features, rank=8, n_bits=4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.n_bits = n_bits
        
        # Pretrained weights (frozen) - will be initialized later
        self.register_buffer('pretrained_weight', 
                           torch.randn(out_features, in_features) * 0.1)
        
        # LoRA parameters with better initialization
        # Using Kaiming-like initialization scaled by rank
        self.lora_A = nn.Parameter(
            torch.randn(rank, in_features) * np.sqrt(2.0 / rank)
        )
        # Initialize B to zero for stable start
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Quantization scales - will be initialized after loading pretrained weights
        self.register_buffer('weight_scale_init', torch.ones(out_features))
        self.register_buffer('activation_scale_init', torch.ones(1))
        
        self.weight_scale = nn.Parameter(torch.ones(out_features))
        self.activation_scale = nn.Parameter(torch.ones(1))
        
        # Bias
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Scaling factor for LoRA (like in the original paper)
        self.scaling = 0.01
        
    def initialize_scales(self):
        """Initialize scales based on pretrained weights"""
        with torch.no_grad():
            # Calculate per-channel statistics for weights
            for i in range(self.out_features):
                channel_weights = self.pretrained_weight[i]
                # Use percentile for more robust initialization
                weight_max = torch.quantile(channel_weights.abs(), 0.99)
                self.weight_scale.data[i] = weight_max / (2**(self.n_bits-1))
            
            # Activation scale - start with a reasonable value
            self.activation_scale.data.fill_(1.0)
            
            # Store initial scales
            self.weight_scale_init = self.weight_scale.data.clone()
            self.activation_scale_init = self.activation_scale.data.clone()
    
    def forward(self, x):
        # Apply LoRA with scaling
        lora_weight = (self.lora_B @ self.lora_A) * self.scaling
        merged_weight = self.pretrained_weight + lora_weight
        
        # Ensure scales are positive and not too small
        weight_scale_clamped = torch.clamp(self.weight_scale.abs(), min=1e-8)
        activation_scale_clamped = torch.clamp(self.activation_scale.abs(), min=1e-8)
        
        # Quantize weights (channel-wise)
        quantized_weights = []
        for i in range(self.out_features):
            w_channel = merged_weight[i:i+1, :]
            w_quant = STEFunction.apply(
                w_channel, 
                weight_scale_clamped[i], 
                torch.zeros(1, device=w_channel.device, dtype=w_channel.dtype),
                self.n_bits
            )
            quantized_weights.append(w_quant)
        
        weight_quantized = torch.cat(quantized_weights, dim=0)
        
        # Quantize activations (layer-wise)
        x_quantized = STEFunction.apply(
            x,
            activation_scale_clamped,
            torch.zeros(1, device=x.device, dtype=x.dtype),
            self.n_bits
        )
        
        # Compute output
        output = F.linear(x_quantized, weight_quantized, self.bias)
        return output

class QALoRAModel(nn.Module):
    """QALoRA model with improved initialization"""
    
    def __init__(self, pretrained_model=None, rank=8, n_bits=4):
        super().__init__()
        
        self.fc1 = QALoRALinear(INPUT_DIM, HIDDEN_DIM, rank, n_bits)
        self.fc2 = QALoRALinear(HIDDEN_DIM, HIDDEN_DIM, rank, n_bits)
        self.fc3 = QALoRALinear(HIDDEN_DIM, OUTPUT_DIM, rank, n_bits)
        self.dropout = nn.Dropout(0.2)
        
        # Initialize from pretrained model if provided
        if pretrained_model is not None:
            self.initialize_from_pretrained(pretrained_model)
    
    def initialize_from_pretrained(self, model):
        """Initialize from pretrained model with proper scale initialization"""
        # Copy pretrained weights
        self.fc1.pretrained_weight.copy_(model.fc1.weight.data)
        self.fc2.pretrained_weight.copy_(model.fc2.weight.data)
        self.fc3.pretrained_weight.copy_(model.fc3.weight.data)
        
        # Copy biases
        self.fc1.bias.data.copy_(model.fc1.bias.data)
        self.fc2.bias.data.copy_(model.fc2.bias.data)
        self.fc3.bias.data.copy_(model.fc3.bias.data)
        
        # Initialize scales based on pretrained weights
        self.fc1.initialize_scales()
        self.fc2.initialize_scales()
        self.fc3.initialize_scales()
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def debug_quantization(model, verbose=True):
    """Debug helper to analyze quantization effects"""
    debug_info = {}
    
    for name, module in model.named_modules():
        if isinstance(module, QALoRALinear):
            with torch.no_grad():
                # Calculate LoRA contribution
                lora_weight = (module.lora_B @ module.lora_A) * module.scaling
                lora_norm = lora_weight.norm().item()
                pretrained_norm = module.pretrained_weight.norm().item()
                
                # Scale statistics
                weight_scale_mean = module.weight_scale.mean().item()
                weight_scale_std = module.weight_scale.std().item()
                activation_scale = module.activation_scale.item()
                
                debug_info[name] = {
                    'lora_norm': lora_norm,
                    'pretrained_norm': pretrained_norm,
                    'lora_ratio': lora_norm / (pretrained_norm + 1e-8),
                    'weight_scale_mean': weight_scale_mean,
                    'weight_scale_std': weight_scale_std,
                    'activation_scale': activation_scale,
                }
                
                if verbose:
                    print(f"\n{name}:")
                    print(f"  LoRA contribution: {lora_norm:.4f} ({lora_norm/pretrained_norm*100:.1f}% of pretrained)")
                    print(f"  Weight scale: {weight_scale_mean:.4f} ± {weight_scale_std:.4f}")
                    print(f"  Activation scale: {activation_scale:.4f}")
    
    return debug_info

def train_qalora_improved():
    print("="*50)
    print("Step 3: QALoRA Implementation (Improved)")
    print("="*50)
    
    # Load pretrained model
    pretrained_model = load_model(BaselineModel, BASELINE_MODEL_PATH)
    
    # Create QALoRA model
    model = QALoRAModel(pretrained_model, rank=LORA_RANK, n_bits=N_BITS)
    model.to(DEVICE)
    
    # Data loaders
    train_loader, test_loader = get_mnist_loaders()
    
    # Separate parameters by type for different learning rates
    lora_params = []
    scale_params = []
    
    for name, param in model.named_parameters():
        if 'lora' in name:
            lora_params.append(param)
            param.requires_grad = True
        elif 'scale' in name:
            scale_params.append(param)
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # Different learning rates for different parameter types
    optimizer = torch.optim.Adam([
        {'params': lora_params, 'lr': LEARNING_RATE},
        {'params': scale_params, 'lr': LEARNING_RATE * 0.1}  # Lower LR for scales
    ])
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    criterion = nn.CrossEntropyLoss()
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in lora_params + scale_params):,}")
    print(f"Percentage trainable: {sum(p.numel() for p in lora_params + scale_params) / sum(p.numel() for p in model.parameters()) * 100:.2f}%")
    
    # Initial evaluation
    initial_accuracy = evaluate_model(model, test_loader)
    print(f"Initial accuracy (before training): {initial_accuracy:.2f}%")
    
    # Training with warmup
    best_accuracy = 0
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        # Warmup: Train only scales for first 2 epochs
        if epoch < 2:
            for param in lora_params:
                param.requires_grad = False
            for param in scale_params:
                param.requires_grad = True
            print(f"Epoch {epoch+1}: Warmup phase - training scales only")
        else:
            for param in lora_params:
                param.requires_grad = True
            for param in scale_params:
                param.requires_grad = True
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(DEVICE).view(-1, INPUT_DIM)
            target = target.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Add L2 regularization for stability
            l2_reg = 0
            for param in lora_params:
                l2_reg += 0.001 * param.norm(2)
            loss = loss + l2_reg
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(lora_params + scale_params, max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 99:
                print(f'Epoch [{epoch+1}/{EPOCHS}], '
                      f'Step [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {running_loss/100:.4f}')
                running_loss = 0.0
        
        # Step scheduler
        scheduler.step()
        
        # Evaluation
        accuracy = evaluate_model(model, test_loader)
        print(f'Epoch [{epoch+1}/{EPOCHS}] Test Accuracy: {accuracy:.2f}%')
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_model(model, QALORA_MODEL_PATH)
            print(f"New best accuracy: {best_accuracy:.2f}%")
        
        # Debug information every few epochs
        if (epoch + 1) % 3 == 0:
            print("\n--- Debug Info ---")
            debug_quantization(model, verbose=False)
    
    # Load best model
    model = load_model(QALoRAModel, QALORA_MODEL_PATH, 
                      pretrained_model=pretrained_model, 
                      rank=LORA_RANK, n_bits=N_BITS)
    
    # Final evaluation
    final_accuracy = evaluate_model(model, test_loader)
    baseline_accuracy = evaluate_model(pretrained_model, test_loader)
    
    print("\n--- Final Comparison ---")
    print(f"Baseline (FP32) Accuracy: {baseline_accuracy:.2f}%")
    print(f"QALoRA ({N_BITS}-bit) Accuracy: {final_accuracy:.2f}%")
    print(f"Accuracy Drop: {baseline_accuracy - final_accuracy:.2f}%")
    
    # Detailed debug info
    print("\n--- Detailed Analysis ---")
    debug_quantization(model, verbose=True)
    
    return model

if __name__ == "__main__":
    model = train_qalora_improved()
    print("\nStep 3 completed successfully!")