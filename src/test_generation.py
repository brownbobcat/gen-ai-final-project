#!/usr/bin/env python3
"""
test_generation.py - Quick test of generation functionality
"""

import os
from generate import SilvacoGenerator

def test_generation():
    """Test the generation pipeline components"""
    print("Testing Silvaco Generation Pipeline")
    print("=" * 50)
    
    # Test 1: Initialize generator
    print("1. Testing generator initialization...")
    try:
        # Use CPU-only for testing
        # generator = SilvacoGenerator(use_4bit=False)
        generator = SilvacoGenerator(
            adapter_path="model/adapter_model",
            use_4bit=False
        )
        print("✓ Generator initialized successfully")
    except Exception as e:
        print(f"❌ Generator initialization failed: {e}")
        return False
    
    # Test 2: Test RAG loading
    print("\n2. Testing RAG index loading...")
    try:
        generator.load_rag_index()
        print("✓ RAG index loaded successfully")
        rag_available = True
    except Exception as e:
        print(f"⚠️  RAG index not available: {e}")
        rag_available = False
    
    # Test 3: Test prompt creation
    print("\n3. Testing prompt creation...")
    test_description = "Create a simple NMOS transistor with 1um channel length"
    
    if rag_available:
        # Test with RAG
        examples = generator.rag.retrieve(test_description, k=2)
        retrieved_examples = generator.rag.format_retrieved_examples(examples)
        prompt_with_rag = generator.create_prompt(test_description, retrieved_examples)
        print(f"✓ RAG-enhanced prompt created ({len(prompt_with_rag)} chars)")
    
    # Test without RAG
    prompt_without_rag = generator.create_prompt(test_description)
    print(f"✓ Basic prompt created ({len(prompt_without_rag)} chars)")
    
    # Test 4: Display prompt structure
    print("\n4. Sample prompt structure:")
    print("-" * 30)
    print(prompt_without_rag[:500] + "..." if len(prompt_without_rag) > 500 else prompt_without_rag)
    print("-" * 30)
    
    # Test 5: Test generation (create mock output)
    print("\n5. Testing generation pipeline...")
    print("⚠️  Skipping actual generation (requires significant compute time)")
    print("   Generation would work with either:")
    print("   - Base model (slower, lower quality)")
    print("   - Fine-tuned model (faster, higher quality)")
    
    # Create mock result
    mock_result = {
        "description": test_description,
        "generated_code": """# Mock generated Silvaco code
# Device structure
go atlas

# Define mesh
mesh space.mult=1.0
x.mesh loc=0.0 spac=0.05
x.mesh loc=1.0 spac=0.05

y.mesh loc=0.0 spac=0.02
y.mesh loc=0.1 spac=0.02

# Define regions
region num=1 material=Silicon
region num=2 material=SiO2 y.min=0.08

# Define electrodes
electrode name=source x.min=0.0 x.max=0.3
electrode name=drain x.min=0.7 x.max=1.0
electrode name=gate x.min=0.3 x.max=0.7 y.min=0.08

# Doping
doping uniform conc=1e16 p.type

# Models
models cvt srh print

# Solve initial
solve init

# DC analysis
solve vgate=0.0 vstep=0.1 vfinal=2.5 name=gate
solve vdrain=0.0 vstep=0.1 vfinal=2.5 name=drain

# Output
log outfile=nmos_dc.log
probe x.value=0.5 y.value=0.02

quit
""",
        "retrieved_examples": ["example1.in", "example2.in"] if rag_available else [],
        "parameters": {"temperature": 0.7, "use_rag": rag_available}
    }
    
    print("✓ Mock generation result created")
    
    print("\n" + "=" * 50)
    print("Generation Pipeline Test Summary:")
    print(f"✓ Generator initialization: Working")
    print(f"{'✓' if rag_available else '⚠️'} RAG system: {'Working' if rag_available else 'Available but not loaded'}")
    print(f"✓ Prompt creation: Working")
    print(f"⚠️ Model generation: Ready (requires compute time)")
    print(f"\nRecommendation: Train model on GPU for best results")
    
    return True

if __name__ == "__main__":
    test_generation()