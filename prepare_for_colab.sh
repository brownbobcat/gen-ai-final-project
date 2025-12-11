#!/bin/bash

# Script to prepare training data for Google Colab upload

echo "Preparing training data for Google Colab..."

# Create a zip file with only the necessary files
zip -r silvaco_training_data.zip \
    data/processed/processed_dataset \
    src/train_qlora.py \
    -x "*.DS_Store" "*__pycache__*"

echo "✓ Created silvaco_training_data.zip"
echo ""
echo "File size:"
ls -lh silvaco_training_data.zip

echo ""
echo "Instructions:"
echo "1. Upload silvaco_training_data.zip to Google Colab"
echo "2. Open train_on_colab.ipynb in Google Colab"
echo "3. Make sure to select GPU runtime (Runtime > Change runtime type > GPU > A100 or T4)"
echo "4. Follow the notebook instructions"
echo ""
echo "The training will take approximately:"
echo "- A100 GPU: 1-1.5 hours"
echo "- T4 GPU: 2-3 hours"