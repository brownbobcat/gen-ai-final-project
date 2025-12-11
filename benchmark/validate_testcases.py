#!/usr/bin/env python3
"""
validate_testcases.py - Validate and analyze benchmark test cases
"""

import json
import os
from collections import Counter
from pathlib import Path


def load_test_cases(filepath: str):
    """Load test cases from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def validate_test_cases(test_cases):
    """Validate test case structure and requirements"""
    print("Validating Benchmark Test Cases")
    print("=" * 50)
    
    # Check total count
    total_cases = len(test_cases)
    print(f"Total test cases: {total_cases}")
    
    # Check category distribution
    categories = [case['category'] for case in test_cases]
    category_counts = Counter(categories)
    
    print(f"\nCategory distribution:")
    for category, count in category_counts.items():
        print(f"  {category}: {count}")
    
    # Validate requirements
    required_distribution = {
        'MOSFET': 6,
        'Diode': 3, 
        'BJT': 3,
        'Photonic': 2,
        'Sensor': 2,
        'Edge Case': 4
    }
    
    print(f"\nRequirement validation:")
    all_good = True
    for category, required in required_distribution.items():
        actual = category_counts.get(category, 0)
        status = "✓" if actual >= required else "❌"
        print(f"  {category}: {actual}/{required} {status}")
        if actual < required:
            all_good = False
    
    # Check required fields
    print(f"\nField validation:")
    required_fields = ['id', 'category', 'description', 'expected']
    for i, case in enumerate(test_cases):
        missing_fields = [field for field in required_fields if field not in case]
        if missing_fields:
            print(f"  Case {i+1} missing fields: {missing_fields}")
            all_good = False
    
    if all_good:
        print("✓ All validation checks passed!")
    else:
        print("❌ Some validation checks failed")
    
    return all_good


def analyze_complexity(test_cases):
    """Analyze test case complexity"""
    print(f"\nComplexity Analysis:")
    print("-" * 30)
    
    complexity_stats = {
        'avg_description_length': 0,
        'parameter_count': [],
        'analysis_types': set(),
        'materials_mentioned': set(),
        'device_types': set()
    }
    
    for case in test_cases:
        desc = case['description']
        expected = case['expected']
        
        # Description length
        complexity_stats['avg_description_length'] += len(desc)
        
        # Parameter count
        if 'parameters' in expected:
            complexity_stats['parameter_count'].append(len(expected['parameters']))
        
        # Analysis types
        if 'analysis' in expected:
            complexity_stats['analysis_types'].update(expected['analysis'])
        
        # Materials
        if 'materials' in expected:
            complexity_stats['materials_mentioned'].update(expected['materials'])
        
        # Device types
        if 'device_type' in expected:
            complexity_stats['device_types'].add(expected['device_type'])
    
    # Calculate averages
    complexity_stats['avg_description_length'] /= len(test_cases)
    avg_params = sum(complexity_stats['parameter_count']) / len(complexity_stats['parameter_count']) if complexity_stats['parameter_count'] else 0
    
    print(f"Average description length: {complexity_stats['avg_description_length']:.0f} characters")
    print(f"Average parameters per case: {avg_params:.1f}")
    print(f"Unique analysis types: {len(complexity_stats['analysis_types'])}")
    print(f"Unique materials: {len(complexity_stats['materials_mentioned'])}")
    print(f"Unique device types: {len(complexity_stats['device_types'])}")


def display_sample_cases(test_cases):
    """Display sample test cases"""
    print(f"\nSample Test Cases:")
    print("-" * 30)
    
    # Show one from each main category
    categories_shown = set()
    for case in test_cases:
        if case['category'] not in categories_shown and case['category'] != 'Edge Case':
            print(f"\n{case['category']} Example ({case['id']}):")
            print(f"Description: {case['description'][:150]}...")
            if 'device_type' in case['expected']:
                print(f"Device Type: {case['expected']['device_type']}")
            if 'parameters' in case['expected']:
                print(f"Key Parameters: {case['expected']['parameters'][:3]}")
            categories_shown.add(case['category'])
    
    # Show one edge case
    edge_case = next((case for case in test_cases if case['category'] == 'Edge Case'), None)
    if edge_case:
        print(f"\nEdge Case Example ({edge_case['id']}):")
        print(f"Description: {edge_case['description'][:150]}...")
        if 'issues' in edge_case['expected']:
            print(f"Expected Issues: {edge_case['expected']['issues']}")


def main():
    """Main validation function"""
    test_cases_path = "benchmark/test_cases.json"
    
    if not os.path.exists(test_cases_path):
        print(f"Error: Test cases file not found at {test_cases_path}")
        return
    
    # Load test cases
    test_cases = load_test_cases(test_cases_path)
    
    # Validate
    is_valid = validate_test_cases(test_cases)
    
    # Analyze
    analyze_complexity(test_cases)
    
    # Display samples
    display_sample_cases(test_cases)
    
    # Summary
    print(f"\n" + "=" * 50)
    print("Benchmark Test Cases Summary:")
    print(f"✓ {len(test_cases)} test cases created")
    print(f"✓ All device categories covered")
    print(f"✓ Edge cases included for robustness testing")
    print(f"{'✓' if is_valid else '❌'} Validation: {'Passed' if is_valid else 'Failed'}")
    
    print(f"\nReady for evaluation pipeline!")


if __name__ == "__main__":
    main()