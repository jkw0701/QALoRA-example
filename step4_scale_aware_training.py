"""
Step 4: Scale-aware LoRA 최적화
레이어별 스케일 차이를 고려한 학습
"""
import torch
import torch.nn as nn
import torch.optim as optim
from config import *
from utils import get_mnist_loaders, evaluate_model, save_model, load_model
from step3_qalora_layer import QALoRAModel
from step1_baseline_model import BaselineModel

class ScaleAwareOptimizer:
    """Scale-aware gradient adjustment"""
    
    def __init__(self, model):
        self.model = model
        
    def adjust_gradients(self):
        """Adjust LoRA gradients based on quantization scales"""
        for name, module in self.model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'weight_scale'):
                if module.lora_A.grad is not None and module.lora_B.grad is not None:
                    # Get average scale for this layer
                    avg_scale = module.weight_scale.mean().detach()
                    
                    # Scale gradients
                    module.lora_A.grad *= avg_scale
                    module.lora_B.grad *= avg_scale

def train_scale_aware():
    print("="*50)
    print("Step 4: Scale-Aware Training")
    print("="*50)
    
    # Load pretrained model
    pretrained_model = load_model(BaselineModel, BASELINE_MODEL_PATH)
    
    # Create two models for comparison
    model_standard = QALoRAModel(pretrained_model, rank=LORA_RANK, n_bits=N_BITS).to(DEVICE)
    model_scale_aware = QALoRAModel(pretrained_model, rank=LORA_RANK, n_bits=N_BITS).to(DEVICE)
    
    # Data loaders
    train_loader, test_loader = get_mnist_loaders()
    
    # Optimizers
    lora_params_standard = [p for n, p in model_standard.named_parameters() 
                           if 'lora' in n or 'scale' in n]
    lora_params_scale_aware = [p for n, p in model_scale_aware.named_parameters() 
                              if 'lora' in n or 'scale' in n]
    
    optimizer_standard = optim.Adam(lora_params_standard, lr=LEARNING_RATE)
    optimizer_scale_aware = optim.Adam(lora_params_scale_aware, lr=LEARNING_RATE)
    
    scale_aware_helper = ScaleAwareOptimizer(model_scale_aware)
    
    criterion = nn.CrossEntropyLoss()
    
    print("Training two models: Standard vs Scale-Aware")
    print("-" * 50)
    
    for epoch in range(EPOCHS):
        # Train standard model
        model_standard.train()
        running_loss_standard = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(DEVICE).view(-1, INPUT_DIM)
            target = target.to(DEVICE)
            
            # Standard training
            optimizer_standard.zero_grad()
            output = model_standard(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer_standard.step()
            running_loss_standard += loss.item()
            
            # Scale-aware training
            optimizer_scale_aware.zero_grad()
            output = model_scale_aware(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Apply scale-aware gradient adjustment
            scale_aware_helper.adjust_gradients()
            
            optimizer_scale_aware.step()
            
            if batch_idx % 200 == 199:
                print(f'Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx+1}], '
                      f'Standard Loss: {running_loss_standard/200:.4f}')
                running_loss_standard = 0.0
        
        # Evaluation
        acc_standard = evaluate_model(model_standard, test_loader)
        acc_scale_aware = evaluate_model(model_scale_aware, test_loader)
        
        print(f'Epoch [{epoch+1}/{EPOCHS}]')
        print(f'  Standard Accuracy: {acc_standard:.2f}%')
        print(f'  Scale-Aware Accuracy: {acc_scale_aware:.2f}%')
        print(f'  Improvement: +{acc_scale_aware - acc_standard:.2f}%')
    
    # Save the better model
    save_model(model_scale_aware, SCALE_AWARE_MODEL_PATH)
    
    # Analyze weight updates
    print("\n--- Analyzing Weight Updates ---")
    for name, module in model_scale_aware.named_modules():
        if hasattr(module, 'lora_B'):
            lora_magnitude = (module.lora_B @ module.lora_A).abs().mean().item()
            scale_mean = module.weight_scale.mean().item()
            print(f"{name}: LoRA magnitude={lora_magnitude:.4f}, "
                  f"Avg scale={scale_mean:.4f}")
    
    return model_scale_aware

if __name__ == "__main__":
    model = train_scale_aware()
    print("\nStep 4 completed successfully!")