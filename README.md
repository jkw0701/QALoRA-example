# EfficientDM Toy Example
A step-by-step implementation of EfficientDM (Efficient Quantization-Aware Fine-Tuning of Low-Bit Diffusion Models) concepts on MNIST dataset.

# 📖 Overview
This repository provides a educational implementation of the key concepts from the paper <ins>"EfficientDM: Efficient Quantization-Aware Fine-Tuning of Low-Bit Diffusion Models" (ICLR 2024)</ins>. While the original paper focuses on diffusion models, this implementation demonstrates the core techniques using a simpler MNIST classification task for clarity and accessibility.

# 🎯 Key Concepts Implemented

- <ins>QALoRA (Quantization-Aware Low-Rank Adaptation)</ins>: Merging LoRA weights with model weights before quantization
- <ins>Scale-Aware Optimization</ins>: Addressing ineffective learning due to varying quantization scales
- <ins>TALSQ (Temporal Activation LSQ)</ins>: Learned step-size quantization for activations
- <ins>Knowledge Distillation</ins>: Transferring knowledge from FP32 to quantized models
- <ins>Data-Free Fine-Tuning</ins>: Training without access to original dataset

# 📂Project Structure
```
efficientdm_toy/
├── config.py                    # Common configuration
├── utils.py                      # Utility functions
├── step1_baseline_model.py      # Train FP32 baseline
├── step2_quantization_basics.py # Basic PTQ implementation
├── step3_qalora_layer.py        # QALoRA implementation
├── step4_scale_aware_training.py # Scale-aware optimization
├── step5_talsq_temporal.py      # TALSQ implementation
├── step6_knowledge_distillation.py # Knowledge distillation
├── step7_full_integration.py    # Final evaluation
├── checkpoints/                 # Saved models
└── data/                        # MNIST dataset
```

# 🚀 Getting Started
## Prerequisites
```
pip install torch torchvision numpy matplotlib
```

## Running the Complete Pipeline
- Execute each step sequentially:
```
# Step 1: Train baseline FP32 model (~98% accuracy)
python step1_baseline_model.py

# Step 2: Apply basic quantization (demonstrates performance degradation)
python step2_quantization_basics.py

# Step 3: Implement QALoRA (recover performance with LoRA)
python step3_qalora_layer.py

# Step 4: Add scale-aware optimization (further improvement)
python step4_scale_aware_training.py

# Step 5: Implement TALSQ (temporal activation scales)
python step5_talsq_temporal.py

# Step 6: Apply knowledge distillation (maximize recovery)
python step6_knowledge_distillation.py

# Step 7: Final evaluation and comparison
python step7_full_integration.py
```

# 📝 License
MIT License - This is an educational implementation for learning purposes.

# 🙏 Acknowledgments
- Original EfficientDM paper authors
- PyTorch team for the framework
- MNIST dataset creators

