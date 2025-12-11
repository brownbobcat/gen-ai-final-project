#!/usr/bin/env python3
"""
generate.py - Generate Silvaco TCAD code using fine-tuned model + RAG
"""

import os
import json
import torch
import argparse
from typing import Optional, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from rag_retrieve import SilvacoRAG


class SilvacoGenerator:
    """Generator for Silvaco TCAD code using model + RAG"""

    def __init__(self,
                 base_model_name: str = "Qwen/Qwen2-0.5B",
                 adapter_path: Optional[str] = None,
                 use_4bit: bool = False):
        """
        Initialize generator
        """
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path or "model/adapter_model"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("=== Generator Initialization ===")
        print(f"Device: {self.device}")
        print(f"Base model: {self.base_model_name}")
        print(f"Adapter path: {self.adapter_path}")
        print(f"Adapter exists? {os.path.isdir(self.adapter_path)}")

        # Load tokenizer
        self.tokenizer = self._load_tokenizer()

        # Load base model
        self.model = self._load_base_model()

        # Load LoRA adapter
        self._load_lora_adapter()

        # Initialize RAG system
        self.rag = SilvacoRAG()

    # -------------------------------------------------------------
    # 1. Load tokenizer
    # -------------------------------------------------------------
    def _load_tokenizer(self):
        print("\n=== Loading tokenizer ===")
        tokenizer_path = "model/tokenizer"
        if os.path.exists(tokenizer_path):
            print(f"Loading tokenizer from {tokenizer_path}")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            print(f"Loading base tokenizer from {self.base_model_name}")
            tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                trust_remote_code=True
            )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    # -------------------------------------------------------------
    # 2. Load base model
    # -------------------------------------------------------------
    def _load_base_model(self):
        print("\n=== Loading base model ===")
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float32,   # FORCE FP32
            trust_remote_code=True,
            device_map=None              # FORCE CPU
        )

        model = model.to("cpu")
        return model
    # -------------------------------------------------------------
    # 3. Load LoRA adapter (robust)
    # -------------------------------------------------------------
    # -------------------------------------------------------------
    # 3. Load LoRA adapter (FULLY FIXED FOR MAC / CPU)
    # -------------------------------------------------------------
    def _load_lora_adapter(self):
        print("\n=== Attempting to load LoRA adapter ===")
        print(f"Adapter path: {self.adapter_path}")

        if not os.path.isdir(self.adapter_path):
            print("⚠️ Adapter folder not found → using base model only")
            return

        try:
            # Load config first
            config = PeftConfig.from_pretrained(self.adapter_path)
            print("✓ Adapter config loaded")

            # Force FP32 for Mac (MPS has NO BF16 support)
            self.model = PeftModel.from_pretrained(
                self.model,
                self.adapter_path,
                is_trainable=False,
                torch_dtype=torch.float32,
                device_map=None  # Force CPU
            )

            # SAFETY: ensure fully on CPU & FP32
            self.model = self.model.to(torch.float32)
            self.model = self.model.to("cpu")

            print("✓ LoRA adapter loaded successfully (FP32 CPU mode)")

        except Exception as e:
            print("⚠️ ERROR loading LoRA adapter:", e)
            print("→ Falling back to base model")

    # -------------------------------------------------------------
    # RAG loading
    # -------------------------------------------------------------
    def load_rag_index(self, index_path="../embeddings/faiss_index.bin",
                       metadata_path="../embeddings/metadata.json"):
        print("\n=== Loading RAG Index ===")
        try:
            self.rag.load_index(index_path, metadata_path)
            print("✓ RAG index loaded successfully")
        except Exception as e:
            print(f"⚠️ Could not load RAG index: {e}")

    # -------------------------------------------------------------
    # Prompt construction
    # -------------------------------------------------------------
    def extract_parameters(self, description):
        """Extract key parameters from device description"""
        import re
        
        params = {}
        
        # Channel length extraction
        length_patterns = [
            r'(\d+\.?\d*)\s*[μµu]m.*(?:channel|gate).*length',
            r'(?:channel|gate).*length.*?(\d+\.?\d*)\s*[μµu]m',
            r'(\d+\.?\d*)\s*nm.*(?:channel|gate).*length',
            r'L\s*=\s*(\d+\.?\d*)\s*[μµu]m'
        ]
        
        for pattern in length_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                unit = 'um' if 'μ' in pattern or 'µ' in pattern or 'u' in pattern else 'nm'
                params['channel_length'] = f"{value}{'u' if unit == 'um' else 'n'}"
                break
        
        # Doping concentration extraction
        doping_patterns = [
            r'(\d+(?:\.\d+)?)\s*[×x]\s*10\^?(\d+).*cm\^?-?3',
            r'10\^?(\d+).*cm\^?-?3',
            r'1e(\d+).*cm\^?-?3'
        ]
        
        for pattern in doping_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                if 'x' in pattern or '×' in pattern:
                    mantissa = float(match.group(1))
                    exponent = int(match.group(2))
                    params['doping'] = f"{mantissa}e{exponent}"
                else:
                    exponent = int(match.group(1))
                    params['doping'] = f"1e{exponent}"
                break
        
        # Analysis type detection
        if 'id-vg' in description.lower() or 'idvg' in description.lower():
            params['analysis'] = 'id_vg_curve'
        elif 'id-vd' in description.lower() or 'idvd' in description.lower():
            params['analysis'] = 'id_vd_curve'
        
        return params
    
    def create_prompt(self, description, retrieved_examples=None):
        # Extract parameters first
        params = self.extract_parameters(description)
        
        prompt = f"""You are a Silvaco ATLAS expert. Generate TCAD simulation code based on the device description.

Device Description: {description}

EXTRACTED PARAMETERS (USE THESE EXACTLY):"""

        if params:
            for key, value in params.items():
                prompt += f"\n- {key}: {value}"
        
        prompt += f"""

REQUIRED Silvaco Structure:
1. go atlas
2. Mesh definition (fine mesh for small devices)
3. Material regions (silicon substrate, oxide if MOSFET)
4. Electrode placement (source, drain, gate)
5. Doping profiles (match specified concentrations)
6. Physical models (srh, auger, fermi for heavy doping)
7. Analysis commands (match requested analysis)
8. quit

Generate complete simulation code using the extracted parameters:
"""
        
        if retrieved_examples:
            prompt += f"\nReference examples:\n{retrieved_examples}\n"
        
        return prompt

    # -------------------------------------------------------------
    # Generation function
    # -------------------------------------------------------------
    def generate(self, description: str, use_rag=True, num_examples=3,
                 max_length=2048, temperature=0.1, top_p=0.95,
                 do_sample=False, repetition_penalty=1.15):

        # Retrieve examples
        retrieved_examples = None
        retrieved_files = []

        if use_rag and self.rag.index is not None:
            print(f"\nRetrieving {num_examples} similar examples...")
            examples = self.rag.retrieve(description, k=num_examples)
            retrieved_examples = self.rag.format_retrieved_examples(examples)
            retrieved_files = [ex["relative_path"] for ex in examples]
            print(f"✓ Retrieved examples: {retrieved_files}")

        # Build prompt
        prompt = self.create_prompt(description, retrieved_examples)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt",
                                truncation=True, max_length=max_length) \
            .to(self.model.device)

        # Determine max_new_tokens  
        input_len = inputs["input_ids"].shape[1]
        max_new_tokens = max(512, min(1024, max_length - input_len))

        print("\nGenerating Silvaco code...")
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=True
            )

        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        generated_code = generated_text[len(prompt):].strip()
        
        # Check if generation is low quality and use template fallback
        if self._is_low_quality_output(generated_code):
            print("⚠️ Low quality output detected, using template fallback")
            template_code = self._generate_template_fallback(description)
            if template_code:
                generated_code = template_code

        return {
            "description": description,
            "generated_code": generated_code,
            "retrieved_examples": retrieved_files,
            "prompt_used": prompt
        }
    
    def _is_low_quality_output(self, text):
        """Check if output is low quality"""
        # Check for comments without actual Silvaco code
        if text.count('#') > text.count('\n') / 3:
            return True
        
        # Check for presence of basic Silvaco commands
        required_commands = ['go atlas', 'mesh', 'region', 'electrode', 'solve', 'quit']
        found_commands = sum(1 for cmd in required_commands if cmd in text.lower())
        
        if found_commands < 4:  # Missing too many essential commands
            return True
            
        return False
    
    def _generate_template_fallback(self, description):
        """Generate template-based code when model fails"""
        try:
            from templates import template_engine
            
            # Use the dynamic template engine directly
            return template_engine.generate_complete_deck(description)
            
        except Exception as e:
            print(f"Template fallback failed: {e}")
            return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str)
    parser.add_argument("--output", "-o", type=str, default="output.in")
    parser.add_argument("--no-rag", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--adapter_path", type=str, default="model/adapter_model")
    args = parser.parse_args()

    generator = SilvacoGenerator(
        adapter_path=args.adapter_path,
        use_4bit=False
    )

    if not args.no_rag:
        generator.load_rag_index()

    description = args.input
    result = generator.generate(
        description,
        use_rag=not args.no_rag,
        temperature=0.1,  # Low temperature for more deterministic output
        max_length=args.max_length,
        do_sample=False,  # Use greedy decoding 
        repetition_penalty=1.2
    )

    # Save output
    with open(args.output, "w") as f:
        f.write(result["generated_code"])

    print("\n=== Generated Silvaco Code ===")
    print(result["generated_code"][:2000])


if __name__ == "__main__":
    main()
