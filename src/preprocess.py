#!/usr/bin/env python3
"""
preprocess.py - Data preprocessing for Silvaco TCAD code generation
Loads JSON dataset, normalizes text, cleans code, and prepares for training
"""

import json
import re
import os
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from tqdm import tqdm


class SilvacoPreprocessor:
    """Preprocesses Silvaco dataset for fine-tuning"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2-0.5B"):
        """Initialize preprocessor with tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def load_json_dataset(self, filepath: str) -> List[Dict]:
        """Load dataset from JSON file"""
        print(f"Loading dataset from {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} examples")
        return data
    
    def normalize_instruction(self, text: str) -> str:
        """Normalize instruction text"""
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        # Normalize unicode characters (μ -> u)
        text = text.replace('μ', 'u')
        # Remove multiple newlines
        text = re.sub(r'\n+', ' ', text)
        return text
    
    def clean_silvaco_code(self, code: str) -> str:
        """Clean and standardize Silvaco code"""
        lines = code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove trailing whitespace
            line = line.rstrip()
            
            # Skip empty lines at the beginning
            if not cleaned_lines and not line:
                continue
            
            # Standardize comment markers
            if line.strip().startswith('*'):
                # Keep comments but standardize format
                cleaned_lines.append(line)
            else:
                # For non-comment lines, ensure consistent spacing
                # Replace tabs with spaces
                line = line.replace('\t', '    ')
                cleaned_lines.append(line)
        
        # Remove trailing empty lines
        while cleaned_lines and not cleaned_lines[-1]:
            cleaned_lines.pop()
        
        return '\n'.join(cleaned_lines)
    
    def create_prompt(self, instruction: str) -> str:
        """Create input prompt for the model"""
        prompt = f"""You are a semiconductor TCAD code generator.

Write a Silvaco ATLAS .in file based on the following device description:

{instruction}

Use correct Silvaco syntax.
Include: structure, mesh, regions, materials, doping, electrodes, models, solve steps.

Silvaco code:"""
        return prompt
    
    def tokenize_examples(self, examples: List[Dict], max_length: int = 2048) -> Dict:
        """Tokenize examples for training"""
        tokenized_inputs = []
        tokenized_labels = []
        
        for example in tqdm(examples, desc="Tokenizing"):
            # Create prompt
            prompt = self.create_prompt(example['instruction'])
            
            # Full text = prompt + output
            full_text = prompt + "\n" + example['output'] + self.tokenizer.eos_token
            
            # Tokenize
            tokenized = self.tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors=None
            )
            
            # Create labels (mask the prompt part)
            prompt_tokens = self.tokenizer(prompt, truncation=True, max_length=max_length)
            prompt_length = len(prompt_tokens['input_ids'])
            
            labels = [-100] * prompt_length + tokenized['input_ids'][prompt_length:]
            labels = labels[:max_length]
            labels = labels + [-100] * (max_length - len(labels))
            
            tokenized_inputs.append({
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask'],
                'labels': labels
            })
        
        # Convert to dataset format
        return {
            'input_ids': [x['input_ids'] for x in tokenized_inputs],
            'attention_mask': [x['attention_mask'] for x in tokenized_inputs],
            'labels': [x['labels'] for x in tokenized_inputs]
        }
    
    def process_dataset(self, input_path: str, output_dir: str, 
                       train_split: float = 0.9, max_length: int = 2048):
        """Full preprocessing pipeline"""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        data = self.load_json_dataset(input_path)
        
        # Process each example
        processed_data = []
        for example in tqdm(data, desc="Processing examples"):
            processed = {
                'instruction': self.normalize_instruction(example['instruction']),
                'output': self.clean_silvaco_code(example['output'])
            }
            processed_data.append(processed)
        
        # Split into train/validation
        split_idx = int(len(processed_data) * train_split)
        train_data = processed_data[:split_idx]
        val_data = processed_data[split_idx:]
        
        print(f"Train examples: {len(train_data)}")
        print(f"Validation examples: {len(val_data)}")
        
        # Tokenize
        print("Tokenizing training data...")
        train_tokenized = self.tokenize_examples(train_data, max_length)
        
        print("Tokenizing validation data...")
        val_tokenized = self.tokenize_examples(val_data, max_length)
        
        # Create HuggingFace datasets
        train_dataset = Dataset.from_dict(train_tokenized)
        val_dataset = Dataset.from_dict(val_tokenized)
        
        # Create dataset dict
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })
        
        # Save to disk
        output_path = os.path.join(output_dir, 'processed_dataset')
        dataset_dict.save_to_disk(output_path)
        print(f"Saved processed dataset to {output_path}")
        
        # Save processed examples for reference
        with open(os.path.join(output_dir, 'processed_examples.json'), 'w') as f:
            json.dump(processed_data[:5], f, indent=2)
        
        # Save dataset info
        info = {
            'total_examples': len(data),
            'train_examples': len(train_data),
            'val_examples': len(val_data),
            'max_length': max_length,
            'model_name': 'Qwen/Qwen2-0.5B'
        }
        with open(os.path.join(output_dir, 'dataset_info.json'), 'w') as f:
            json.dump(info, f, indent=2)
        
        return dataset_dict


def main():
    """Main preprocessing function"""
    # Paths
    input_path = 'data/silvaco_dataset_train.json'
    output_dir = 'data/processed'
    
    # Initialize preprocessor
    preprocessor = SilvacoPreprocessor(model_name="Qwen/Qwen2-0.5B")
    
    # Process dataset
    dataset = preprocessor.process_dataset(
        input_path=input_path,
        output_dir=output_dir,
        train_split=0.9,
        max_length=2048
    )
    
    print("\nPreprocessing complete!")
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['validation'])}")


if __name__ == "__main__":
    main()