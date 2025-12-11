#!/usr/bin/env python3
"""
test_enhanced_system.py - Test the enhanced Few-Shot prompting system
"""

import os
import sys

# Add src to path
sys.path.append('src')

from enhanced_prompts import create_enhanced_prompt, enhanced_prompts


def test_enhanced_prompts():
    """Test the enhanced prompting system"""
    print("=== TESTING ENHANCED TCAD PROMPTING SYSTEM ===\n")
    
    # Test cases covering different device types and complexities
    test_cases = [
        {
            'description': 'Create a basic NMOS transistor with 1μm channel length and 10μm width for DC analysis',
            'expected_params': {'L': '1u', 'W': '10u', 'sim_type': 'auto'}
        },
        {
            'description': 'Design a PMOS device with 0.5μm gate length, 20μm width, 1.8V supply for Id-Vd curves',
            'expected_params': {'L': '0.5u', 'W': '20u', 'vdd': '1.8V', 'sim_type': 'Id-Vd'}
        },
        {
            'description': 'Create a 180nm NMOS with 3nm oxide thickness for quantum effects analysis',
            'expected_params': {'L': '180n', 'tox': '3n', 'sim_type': 'auto'}
        },
        {
            'description': 'Design a power MOSFET with 10μm channel length, 1000μm width, 20V operation',
            'expected_params': {'L': '10u', 'W': '1000u', 'vdd': '20V'}
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"=== TEST CASE {i} ===")
        description = test_case['description']
        print(f"Description: {description}")
        
        # Test parameter extraction
        params = enhanced_prompts.extract_parameters(description)
        device_type = enhanced_prompts.detect_device_type(description)
        
        print(f"Detected device type: {device_type}")
        print(f"Extracted parameters: {params}")
        
        # Check if key parameters were extracted correctly
        expected = test_case['expected_params']
        print("Parameter extraction validation:")
        for key, expected_value in expected.items():
            actual_value = params.get(key, 'not found')
            status = "✓" if actual_value == expected_value else "✗"
            print(f"  {key}: expected '{expected_value}', got '{actual_value}' {status}")
        
        # Generate enhanced prompt
        prompt = create_enhanced_prompt(description)
        print(f"Prompt length: {len(prompt)} characters")
        print(f"Prompt contains examples: {'Example 1' in prompt and 'Example 2' in prompt}")
        print(f"Prompt contains rules: {'Follow the rules:' in prompt}")
        print(f"Prompt contains structure guide: {'OUTPUT FORMAT' in prompt}")
        
        # Show first 200 chars of prompt for verification
        print(f"Prompt preview: {prompt[:200]}...")
        print("-" * 80)
        print()
    
    print("=== TESTING COMPLETE ===")
    print("Enhanced prompting system ready for use!")


if __name__ == "__main__":
    test_enhanced_prompts()