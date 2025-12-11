#!/usr/bin/env python3
"""
generate.py - Generate SPICE netlists using the fine-tuned model + template fallback.
"""

import argparse
import os
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from enhanced_prompts import create_enhanced_prompt
from templates import template_engine

FORBIDDEN_KEYWORDS = [
    "go atlas",
    "go victory",
    "mesh",
    "region",
    "deposit",
    "etch",
    "oxidize",
    "remesh",
    "electrode",
    "structure=",
]


class SPICEModel:
    """Wrapper around the fine-tuned language model to generate SPICE netlists."""

    def __init__(self, base_model: str, adapter_path: Optional[str] = None):
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        if adapter_path and os.path.isdir(adapter_path):
            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_path,
                torch_dtype=dtype,
                is_trainable=False,
            ).to(self.model.device)

        self.model.eval()

    def generate(self, description: str, max_new_tokens: int = 600, temperature: float = 0.2) -> str:
        """Generate SPICE code from a natural-language description."""
        prompt = create_enhanced_prompt(description)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
            )

        raw = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "### NOW GENERATE THE SPICE NETLIST:" in raw:
            raw = raw.split("### NOW GENERATE THE SPICE NETLIST:")[-1]

        # Clean up the output by removing repetitive sections
        spice_code = raw.strip()
        
        # Find the first valid SPICE netlist (starting with a comment or component)
        lines = spice_code.split('\n')
        start_idx = 0
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('*') or line.startswith(('V', 'R', 'C', 'L', 'M', 'Q', 'I')) or line.startswith('.'):
                if not line.startswith('*** '):  # Skip our template markers
                    start_idx = i
                    break
        
        # Find the first .END
        end_idx = len(lines)
        for i in range(start_idx, len(lines)):
            if lines[i].strip().upper() == '.END':
                end_idx = i + 1
                break
        
        spice_code = '\n'.join(lines[start_idx:end_idx])
        
        if not spice_code.lower().endswith(".end"):
            spice_code = spice_code.rstrip() + "\n.END"

        # Safety filter with template fallback
        if any(term in spice_code.lower() for term in FORBIDDEN_KEYWORDS):
            print("⚠ Model output contaminated → using template engine fallback")
            spice_code = template_engine.generate_complete_netlist(description)

        return spice_code.strip()


def main():
    parser = argparse.ArgumentParser(description="Generate SPICE netlists from natural language.")
    parser.add_argument("--input", "-i", required=True, help="Device description to implement.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2-0.5B",
        help="Base Hugging Face model to load."
    )
    parser.add_argument(
        "--adapter_path",
        default="model/adapter_model",
        help="Path to LoRA adapter (optional)."
    )
    parser.add_argument(
        "--output",
        "-o",
        default="output.sp",
        help="Output file for the generated SPICE deck."
    )
    parser.add_argument("--max-new-tokens", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    generator = SPICEModel(args.model, adapter_path=args.adapter_path)
    netlist = generator.generate(
        args.input,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    with open(args.output, "w") as f:
        f.write(netlist + "\n")

    print("\n=== Generated SPICE Netlist ===")
    print(netlist[:2000])


if __name__ == "__main__":
    main()
