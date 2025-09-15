"""
Step 6: Knowledge Distillation
FP32 모델의 지식을 양자화 모델로 전달
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from utils import get_mnist_loaders, evaluate_model, save_model, load_model
from step1_baseline_model import BaselineModel
from step5_talsq_temporal import TALSQModel
import numpy as np

class DistillationLoss(nn.Module):
    """Knowledge distillation loss"""
    
    def __init__(self, alpha=0.7, temperature=3.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, student_logits, teacher_logits, targets):
        # Soft targets loss
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            soft_targets,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Hard targets loss
        hard_loss = self.ce_loss(student_logits, targets)
        
        # Combined loss
        loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        return loss, soft_loss, hard_loss

def train_with_distillation():
    print("="*50)
    print("Step 6: Knowledge Distillation")
    print("="*50)
    
    # Load teacher model (FP32)
    teacher_model = load_model(BaselineModel, BASELINE_MODEL_PATH)
    teacher_model.eval()  # Teacher always in eval mode
    
    # Create student model (Quantized)
    student_model = TALSQModel(teacher_model, rank=LORA_RANK, n_bits=N_BITS, n_steps=10)
    student_model.to(DEVICE)
    
    # Data loaders
    train_loader, test_loader = get_mnist_loaders()
    
    # Loss and optimizer
    distillation_loss = DistillationLoss(alpha=0.7, temperature=3.0)
    
    params = [p for n, p in student_model.named_parameters() 
             if 'lora' in n or 'scale' in n or 'temporal' in n]
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)
    
    print("Training with knowledge distillation")
    print(f"Teacher model: FP32")
    print(f"Student model: {N_BITS}-bit quantized")
    print("-" * 50)
    
    # Training
    for epoch in range(EPOCHS):
        student_model.train()
        running_loss = 0.0
        running_soft_loss = 0.0
        running_hard_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(DEVICE).view(-1, INPUT_DIM)
            target = target.to(DEVICE)
            
            # Set temporal step
            step = batch_idx % student_model.n_steps
            student_model.set_step(step)
            
            # Get teacher predictions
            with torch.no_grad():
                teacher_logits = teacher_model(data)
            
            # Get student predictions
            student_logits = student_model(data)
            
            # Calculate distillation loss
            loss, soft_loss, hard_loss = distillation_loss(
                student_logits, teacher_logits, target
            )
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_soft_loss += soft_loss.item()
            running_hard_loss += hard_loss.item()
            
            if batch_idx % 100 == 99:
                print(f'Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx+1}]')
                print(f'  Total Loss: {running_loss/100:.4f}')
                print(f'  Soft Loss: {running_soft_loss/100:.4f}, '
                      f'Hard Loss: {running_hard_loss/100:.4f}')
                running_loss = 0.0
                running_soft_loss = 0.0
                running_hard_loss = 0.0
        
        # Evaluation
        teacher_acc = evaluate_model(teacher_model, test_loader)
        
        # Average accuracy across temporal steps
        accuracies = []
        for step in range(student_model.n_steps):
            student_model.set_step(step)
            acc = evaluate_model(student_model, test_loader)
            accuracies.append(acc)
        student_acc = np.mean(accuracies)
        
        print(f'Epoch [{epoch+1}/{EPOCHS}] Results:')
        print(f'  Teacher Accuracy: {teacher_acc:.2f}%')
        print(f'  Student Accuracy: {student_acc:.2f}%')
        print(f'  Gap: {teacher_acc - student_acc:.2f}%')
    
    # Save final model
    save_model(student_model, FINAL_MODEL_PATH)
    
    return student_model

def compare_outputs():
    """Compare teacher and student outputs"""
    print("\n--- Output Comparison ---")
    
    teacher = load_model(BaselineModel, BASELINE_MODEL_PATH)
    student = load_model(TALSQModel, FINAL_MODEL_PATH, 
                        rank=LORA_RANK, n_bits=N_BITS, n_steps=10)
    
    teacher.eval()
    student.eval()
    
    # Get sample data
    _, test_loader = get_mnist_loaders()
    data, target = next(iter(test_loader))
    data = data[:5].to(DEVICE).view(5, -1)
    
    with torch.no_grad():
        teacher_out = F.softmax(teacher(data), dim=1)
        student.set_step(0)
        student_out = F.softmax(student(data), dim=1)
    
    print("Sample predictions (top-3 probabilities):")
    for i in range(5):
        print(f"\nSample {i+1}:")
        
        # Teacher
        teacher_probs, teacher_indices = torch.topk(teacher_out[i], 3)
        print(f"  Teacher: ", end="")
        for prob, idx in zip(teacher_probs, teacher_indices):
            print(f"{idx.item()}:{prob.item():.3f} ", end="")
        
        # Student
        print()
        student_probs, student_indices = torch.topk(student_out[i], 3)
        print(f"  Student: ", end="")
        for prob, idx in zip(student_probs, student_indices):
            print(f"{idx.item()}:{prob.item():.3f} ", end="")
        print()

if __name__ == "__main__":
    model = train_with_distillation()
    compare_outputs()
    print("\nStep 6 completed successfully!")