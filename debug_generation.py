#!/usr/bin/env python3
"""
debug_generation.py - Debug the generation issue
"""

import sys
import os
sys.path.append('src')

from generate import SilvacoGenerator

def debug_generation():
    """Debug generation with simple input"""
    
    print("=== DEBUG GENERATION ===")
    
    # Initialize generator
    generator = SilvacoGenerator(
        adapter_path="model/adapter_model",
        use_4bit=False
    )
    
    # Load RAG
    generator.load_rag_index("embeddings/faiss_index.bin", "embeddings/metadata.json")
    
    # Simple test input
    description = "Create a basic NMOS with 1μm channel length for testing"
    
    print(f"Input description: {description}")
    
    # Generate with debug info
    result = generator.generate(
        description,
        use_rag=True,
        temperature=0.1,
        max_length=2048,
        do_sample=False,
        repetition_penalty=1.2
    )
    
    print(f"\nGenerated code length: {len(result['generated_code'])}")
    print(f"Generated code: '{result['generated_code']}'")
    print(f"\nPrompt used length: {len(result['prompt_used'])}")
    print(f"Prompt preview: {result['prompt_used'][:200]}...")
    
    # Try with custom prompt method
    print("\n=== TESTING CUSTOM PROMPT METHOD ===")
    from enhanced_prompts import create_enhanced_prompt
    custom_prompt = create_enhanced_prompt(description)
    
    custom_result = generator.generate_with_custom_prompt(
        custom_prompt,
        max_length=2048,
        temperature=0.1,
        do_sample=False
    )
    
    print(f"Custom prompt result length: {len(custom_result)}")
    print(f"Custom prompt result: '{custom_result[:500]}...'")

if __name__ == "__main__":
    debug_generation()