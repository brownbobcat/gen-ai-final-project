#!/usr/bin/env python3
"""
Simple test to check generation output
"""

import subprocess
import sys
import os

def test_generation():
    """Test the generation and capture output"""
    
    print("=== TESTING GENERATION OUTPUT ===")
    
    # Change to src directory and run generation
    cmd = [
        "python3", "generate.py",
        "--input", "Create a basic NMOS with 1μm channel length for testing",
        "--adapter_path", "../model/adapter_model",
        "--output", "test_output.in"
    ]
    
    try:
        # Set working directory to src
        result = subprocess.run(
            cmd, 
            cwd="src",
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        print("STDOUT:")
        print(result.stdout)
        print("\nSTDERR:")
        print(result.stderr)
        print(f"\nReturn code: {result.returncode}")
        
        # Check if output file was created
        output_file = "src/test_output.in"
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                content = f.read()
            print(f"\nOutput file length: {len(content)}")
            print(f"Output file content: '{content[:500]}'")
        else:
            print("\nNo output file created")
            
    except subprocess.TimeoutExpired:
        print("Generation timed out after 5 minutes")
    except Exception as e:
        print(f"Error running generation: {e}")

if __name__ == "__main__":
    test_generation()