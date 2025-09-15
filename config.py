"""
공통 설정 파일
"""
import os
import torch

# 경로 설정
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, 'checkpoints')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

# 디렉토리 생성
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# 하이퍼파라미터
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 설정
INPUT_DIM = 784  # MNIST flattened
HIDDEN_DIM = 256
OUTPUT_DIM = 10
LORA_RANK = 8
N_BITS = 4

# 파일 경로
BASELINE_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'baseline_model.pth')
QUANTIZED_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'quantized_model.pth')
QALORA_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'qalora_model.pth')
SCALE_AWARE_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'scale_aware_model.pth')
TALSQ_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'talsq_model.pth')
FINAL_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'final_model.pth')

print(f"Project directory: {PROJECT_DIR}")
print(f"Checkpoint directory: {CHECKPOINT_DIR}")
print(f"Device: {DEVICE}")