#!/usr/bin/env python3
"""
test_model_generation.py - Test actual model generation with strengthened SPICE prompts
"""

import sys
import os

def test_model_generation():
    """Test the actual model generation with the 4 validation queries"""
    
    print("=== TESTING MODEL GENERATION WITH STRENGTHENED SPICE PROMPTS ===\n")
    
    try:
        from generate import SilvacoGenerator
        
        # Initialize generator
        print("Initializing generator...")
        generator = SilvacoGenerator(use_4bit=False)
        
        # Load RAG if available
        try:
            generator.load_rag_index()
            print("✓ RAG system loaded")
        except Exception as e:
            print(f"⚠️ RAG not available: {e}")
        
        # Test queries that previously failed
        test_queries = [
            "Generate a SPICE netlist of a 45 nm NMOS and sweep Id–Vg.",
            "Create a BJT Gummel plot circuit using .DC sweep.",
            "Build a MOS capacitor and run CV analysis.",
            "Generate a PMOS device analysis"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*60}")
            print(f"TEST {i}: {query}")
            print('='*60)
            
            try:
                # Generate with strengthened SPICE prompts
                result = generator.generate(
                    query,
                    use_rag=True,
                    temperature=0.1,
                    do_sample=False,
                    repetition_penalty=1.3
                )
                
                generated = result["generated_code"]
                
                print(f"\nGenerated output ({len(generated)} chars):")
                print("-" * 40)
                print(generated)
                print("-" * 40)
                
                # Analyze output quality
                if "ERROR: Model generated invalid output" in generated:
                    print("❌ VALIDATION FAILED: Model output rejected by quality check")
                elif any(atlas_cmd in generated.lower() for atlas_cmd in ['go atlas', 'mesh', 'region', 'electrode', 'solve', 'quit']):
                    print("❌ VALIDATION FAILED: ATLAS contamination detected")
                elif generated.count('.MODEL') > 5:
                    print("❌ VALIDATION FAILED: Excessive repetition detected")
                elif '.END' not in generated.upper():
                    print("❌ VALIDATION FAILED: Missing .END statement")
                elif not any(spice_elem in generated.upper() for spice_elem in ['M1', 'Q1', '.DC', '.AC', '.TRAN']):
                    print("❌ VALIDATION FAILED: Missing essential SPICE elements")
                else:
                    print("✅ VALIDATION PASSED: Clean SPICE netlist generated")
                
            except Exception as e:
                print(f"❌ GENERATION FAILED: {e}")
        
        print(f"\n{'='*60}")
        print("TESTING COMPLETE")
        print('='*60)
        
    except ImportError as e:
        print(f"❌ Failed to import dependencies: {e}")
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_model_generation()