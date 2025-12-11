#!/usr/bin/env python3
"""
GPU Training Script - Run this on a GPU-enabled system
"""

import torch
from train_qlora import main

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        main()
    else:
        print("No GPU detected. Please run on a GPU-enabled system.")
