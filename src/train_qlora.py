#!/usr/bin/env python3
"""
train_qlora.py - Fine-tune Qwen model using QLoRA for Silvaco code generation
Uses 4-bit quantization and LoRA adapters for efficient training
"""

import os
import json
import torch
from pathlib import Path
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import numpy as np
from tqdm import tqdm


class SilvacoQLoRATrainer:
    """QLoRA trainer for Silvaco code generation"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2-0.5B"):
        """Initialize trainer with model configuration"""
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # QLoRA configuration
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        # LoRA configuration
        self.lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.model = None
        self.tokenizer = None
        
    def setup_model(self):
        """Load and prepare model for QLoRA training"""
        print(f"Loading base model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with 4-bit quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Add LoRA adapters
        self.model = get_peft_model(self.model, self.lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        
        # Shift predictions and labels for next token prediction
        predictions = predictions[:, :-1, :]
        labels = labels[:, 1:]
        
        # Calculate perplexity
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        losses = []
        
        for i in range(0, len(predictions), 8):  # Process in batches
            batch_preds = torch.tensor(predictions[i:i+8])
            batch_labels = torch.tensor(labels[i:i+8])
            
            batch_losses = loss_fct(
                batch_preds.reshape(-1, batch_preds.size(-1)),
                batch_labels.reshape(-1)
            )
            
            # Mask out padding
            mask = (batch_labels != -100).float()
            batch_losses = batch_losses.reshape(batch_labels.shape)
            
            # Average loss per sequence
            seq_losses = (batch_losses * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            losses.extend(seq_losses.tolist())
        
        perplexity = np.exp(np.mean(losses))
        
        return {"perplexity": perplexity}
    
    def train(self, train_dataset_path: str, output_dir: str, 
              num_epochs: int = 3, batch_size: int = 8, 
              learning_rate: float = 2e-4, warmup_steps: int = 100):
        """Train the model"""
        
        # Load dataset
        print(f"Loading dataset from {train_dataset_path}")
        dataset = load_from_disk(train_dataset_path)
        
        # Setup model if not already done
        if self.model is None:
            self.setup_model()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=warmup_steps,
            learning_rate=learning_rate,
            fp16=False,  # Use bf16 instead
            bf16=True,
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="perplexity",
            greater_is_better=False,
            push_to_hub=False,
            report_to=["tensorboard"],
            logging_dir=f"{output_dir}/logs",
            remove_unused_columns=False,
            label_names=["labels"],
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        # Train
        print("Starting training...")
        train_result = trainer.train()
        
        # Save model
        print("Saving model...")
        trainer.save_model(f"{output_dir}/final_model")
        
        # Save training results
        with open(f"{output_dir}/train_results.json", "w") as f:
            json.dump(train_result.metrics, f, indent=2)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(f"{output_dir}/tokenizer")
        
        # Save adapter config
        self.model.save_pretrained(f"{output_dir}/adapter_model")
        
        print(f"Training complete! Model saved to {output_dir}")
        
        return train_result
    
    def inference_test(self, prompt: str, max_length: int = 512):
        """Test the model with a sample prompt"""
        if self.model is None:
            raise ValueError("Model not loaded. Call setup_model() or train() first.")
        
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated


def main():
    """Main training function"""
    # Configuration
    model_name = "Qwen/Qwen2-0.5B"
    dataset_path = "data/processed/processed_dataset"
    output_dir = "model"
    
    # Training hyperparameters
    num_epochs = 3
    batch_size = 8
    learning_rate = 2e-4
    
    # Initialize trainer
    trainer = SilvacoQLoRATrainer(model_name)
    
    # Check if GPU is available
    if not torch.cuda.is_available():
        print("WARNING: GPU not available. Training will be very slow on CPU.")
        print("Consider using a cloud GPU service or reducing model size.")
        
        # For CPU, use smaller batch size
        batch_size = 2
    
    # Train model
    trainer.train(
        train_dataset_path=dataset_path,
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    
    # Test inference
    print("\n" + "="*50)
    print("Testing model inference...")
    test_prompt = """You are a semiconductor TCAD code generator.

Write a Silvaco ATLAS .in file based on the following device description:

Create a simple NMOS transistor with 1um channel length and 10um width, operating at 3V supply voltage.

Use correct Silvaco syntax.
Include: structure, mesh, regions, materials, doping, electrodes, models, solve steps.

Silvaco code:"""
    
    generated = trainer.inference_test(test_prompt, max_length=512)
    print(f"\nGenerated output:\n{generated}")


if __name__ == "__main__":
    main()