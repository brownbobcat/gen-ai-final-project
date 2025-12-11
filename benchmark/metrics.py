#!/usr/bin/env python3
"""
metrics.py - Evaluation metrics for Silvaco TCAD code generation
Implements SVS, PEM, CCS, and BLEU metrics as required
"""

import re
import json
import math
from typing import Dict, List, Tuple, Optional
from collections import Counter
import numpy as np


class SilvacoMetrics:
    """Evaluation metrics for Silvaco TCAD code generation"""
    
    def __init__(self):
        """Initialize metrics calculator"""
        # Required Silvaco sections for syntax validation
        self.required_sections = [
            'go atlas',      # Start Atlas
            'mesh',          # Mesh definition
            'region',        # Device regions
            'electrode',     # Electrodes/contacts
            'models',        # Physical models
            'solve',         # Solution commands
            'quit'           # End simulation
        ]
        
        # Component categories for CCS
        self.component_categories = {
            'structure': ['mesh', 'region', 'material'],
            'electrodes': ['electrode', 'contact'],
            'parameters': ['doping', 'material', 'models'],
            'analysis': ['solve', 'log', 'save'],
            'models': ['models', 'material', 'method'],
            'output': ['log', 'save', 'extract', 'quit']
        }
    
    def syntax_validity_score(self, generated_code: str) -> Dict[str, any]:
        """
        Syntax Validity Score (SVS)
        Detects presence of required Silvaco sections
        Returns 1 if all required sections present, 0 otherwise
        """
        code_lower = generated_code.lower()
        
        # Check for each required section
        sections_found = []
        sections_missing = []
        
        for section in self.required_sections:
            if section in code_lower:
                sections_found.append(section)
            else:
                sections_missing.append(section)
        
        # Calculate score (binary: all or nothing)
        score = 1.0 if len(sections_missing) == 0 else 0.0
        
        # Also calculate partial score for analysis
        partial_score = len(sections_found) / len(self.required_sections)
        
        return {
            'svs_score': score,
            'partial_svs': partial_score,
            'sections_found': sections_found,
            'sections_missing': sections_missing,
            'total_required': len(self.required_sections),
            'found_count': len(sections_found)
        }
    
    def parameter_exact_match(self, generated_code: str, expected_params: List[str]) -> Dict[str, any]:
        """
        Parameter Exact Match (PEM)
        Extract parameters using regex and compare to expected
        """
        if not expected_params:
            return {'pem_score': 1.0, 'matched_params': [], 'missing_params': [], 'extracted_params': []}
        
        # Parameter extraction patterns
        param_patterns = {
            'dimensions': r'(?:L|W|length|width)\s*=\s*([0-9.]+[unm]?)',
            'voltages': r'(?:V\w*|voltage)\s*=\s*([0-9.-]+[V]?)',
            'doping': r'(?:doping|conc)\s*=\s*([0-9.e+-]+)',
            'temperature': r'(?:temp|temperature)\s*=\s*([0-9.-]+)',
            'frequency': r'(?:freq|frequency)\s*=\s*([0-9.]+[GMK]?[Hh]z)',
            'thickness': r'(?:thickness|tox)\s*=\s*([0-9.]+[unm]?)',
            'materials': r'(?:material)\s*=\s*([A-Za-z0-9]+)',
            'models': r'(?:models|model)\s+([A-Za-z0-9\s]+)',
            'spacing': r'(?:spac|spacing)\s*=\s*([0-9.]+[unm]?)'
        }
        
        # Extract parameters from generated code
        extracted_params = []
        code_lower = generated_code.lower()
        
        for param_type, pattern in param_patterns.items():
            matches = re.findall(pattern, code_lower, re.IGNORECASE)
            for match in matches:
                extracted_params.append(f"{param_type}={match}")
        
        # Also look for direct parameter matches
        direct_patterns = [
            r'([LW])\s*=\s*([0-9.]+[unm]?)',          # L=1u, W=10u
            r'(VDD|VCC|Vdd)\s*=\s*([0-9.-]+[V]?)',    # VDD=3V
            r'(\w+)\s*=\s*([0-9.e+-]+[unm]?[V]?)',    # Generic param=value
        ]
        
        for pattern in direct_patterns:
            matches = re.findall(pattern, code_lower, re.IGNORECASE)
            for param, value in matches:
                extracted_params.append(f"{param}={value}")
        
        # Match against expected parameters
        matched_params = []
        missing_params = []
        
        for expected in expected_params:
            expected_lower = expected.lower()
            found = False
            
            # Direct string match
            if expected_lower in code_lower:
                matched_params.append(expected)
                found = True
            else:
                # Fuzzy parameter matching
                if '=' in expected_lower:
                    param_name, param_value = expected_lower.split('=', 1)
                    param_name = param_name.strip()
                    param_value = param_value.strip()
                    
                    # Look for parameter name and approximate value
                    param_pattern = f"{param_name}\\s*=\\s*{re.escape(param_value)}"
                    if re.search(param_pattern, code_lower, re.IGNORECASE):
                        matched_params.append(expected)
                        found = True
                    else:
                        # Try without units
                        param_value_no_units = re.sub(r'[a-zA-Z]', '', param_value)
                        if param_value_no_units:
                            param_pattern = f"{param_name}\\s*=\\s*{re.escape(param_value_no_units)}"
                            if re.search(param_pattern, code_lower, re.IGNORECASE):
                                matched_params.append(expected)
                                found = True
            
            if not found:
                missing_params.append(expected)
        
        # Calculate score
        score = len(matched_params) / len(expected_params) if expected_params else 1.0
        
        return {
            'pem_score': score,
            'matched_params': matched_params,
            'missing_params': missing_params,
            'extracted_params': list(set(extracted_params)),
            'total_expected': len(expected_params),
            'matched_count': len(matched_params)
        }
    
    def component_completeness_score(self, generated_code: str) -> Dict[str, any]:
        """
        Component Completeness Score (CCS)
        Check presence of essential simulation components
        """
        code_lower = generated_code.lower()
        
        # Check each component category
        category_scores = {}
        found_components = {}
        
        for category, keywords in self.component_categories.items():
            found_in_category = []
            
            for keyword in keywords:
                if keyword in code_lower:
                    found_in_category.append(keyword)
            
            # Score for this category (binary: at least one keyword found)
            category_scores[category] = 1.0 if found_in_category else 0.0
            found_components[category] = found_in_category
        
        # Overall CCS score (average of category scores)
        ccs_score = sum(category_scores.values()) / len(category_scores)
        
        # Count essential components
        essential_components = ['structure', 'electrodes', 'analysis']
        essential_score = sum(category_scores[comp] for comp in essential_components) / len(essential_components)
        
        return {
            'ccs_score': ccs_score,
            'essential_score': essential_score,
            'category_scores': category_scores,
            'found_components': found_components,
            'total_categories': len(self.component_categories),
            'categories_found': sum(1 for score in category_scores.values() if score > 0)
        }
    
    def bleu_similarity(self, generated_code: str, reference_code: str) -> Dict[str, any]:
        """
        Optional BLEU similarity score
        Computes BLEU score between generated and reference code
        """
        def tokenize(text: str) -> List[str]:
            """Simple tokenization for code"""
            # Split on whitespace and common separators
            tokens = re.split(r'[\s=\(\)\[\],;]+', text.lower())
            return [token for token in tokens if token and len(token) > 0]
        
        def ngrams(tokens: List[str], n: int) -> List[Tuple]:
            """Generate n-grams from tokens"""
            return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        
        def bleu_n(generated_tokens: List[str], reference_tokens: List[str], n: int) -> float:
            """Compute BLEU-n score"""
            if len(generated_tokens) < n:
                return 0.0
            
            generated_ngrams = ngrams(generated_tokens, n)
            reference_ngrams = ngrams(reference_tokens, n)
            
            if not generated_ngrams:
                return 0.0
            
            generated_counts = Counter(generated_ngrams)
            reference_counts = Counter(reference_ngrams)
            
            matches = sum(min(generated_counts[ngram], reference_counts[ngram]) 
                         for ngram in generated_counts)
            
            return matches / len(generated_ngrams)
        
        # Tokenize
        gen_tokens = tokenize(generated_code)
        ref_tokens = tokenize(reference_code)
        
        if not gen_tokens or not ref_tokens:
            return {'bleu_score': 0.0, 'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0}
        
        # Compute BLEU-n scores
        bleu_1 = bleu_n(gen_tokens, ref_tokens, 1)
        bleu_2 = bleu_n(gen_tokens, ref_tokens, 2) 
        bleu_3 = bleu_n(gen_tokens, ref_tokens, 3)
        bleu_4 = bleu_n(gen_tokens, ref_tokens, 4)
        
        # Brevity penalty
        gen_len = len(gen_tokens)
        ref_len = len(ref_tokens)
        
        if gen_len > ref_len:
            bp = 1.0
        else:
            bp = math.exp(1 - ref_len / gen_len) if gen_len > 0 else 0.0
        
        # Overall BLEU score (geometric mean of BLEU-1 to BLEU-4)
        if bleu_1 > 0 and bleu_2 > 0 and bleu_3 > 0 and bleu_4 > 0:
            bleu_score = bp * math.exp((math.log(bleu_1) + math.log(bleu_2) + 
                                      math.log(bleu_3) + math.log(bleu_4)) / 4)
        else:
            bleu_score = 0.0
        
        return {
            'bleu_score': bleu_score,
            'bleu_1': bleu_1,
            'bleu_2': bleu_2, 
            'bleu_3': bleu_3,
            'bleu_4': bleu_4,
            'brevity_penalty': bp,
            'generated_length': gen_len,
            'reference_length': ref_len
        }
    
    def evaluate_all(self, generated_code: str, expected_params: Optional[List[str]] = None, 
                    reference_code: Optional[str] = None) -> Dict[str, any]:
        """
        Compute all evaluation metrics
        """
        results = {}
        
        # SVS Score
        svs_results = self.syntax_validity_score(generated_code)
        results.update(svs_results)
        
        # PEM Score
        if expected_params:
            pem_results = self.parameter_exact_match(generated_code, expected_params)
            results.update(pem_results)
        else:
            results.update({
                'pem_score': 1.0,
                'matched_params': [],
                'missing_params': [],
                'extracted_params': []
            })
        
        # CCS Score
        ccs_results = self.component_completeness_score(generated_code)
        results.update(ccs_results)
        
        # BLEU Score (optional)
        if reference_code:
            bleu_results = self.bleu_similarity(generated_code, reference_code)
            results.update(bleu_results)
        
        # Overall composite score
        composite_score = (results['svs_score'] + results['pem_score'] + results['ccs_score']) / 3
        results['composite_score'] = composite_score
        
        return results


def test_metrics():
    """Test the metrics with sample data"""
    print("Testing Silvaco Evaluation Metrics")
    print("=" * 50)
    
    metrics = SilvacoMetrics()
    
    # Sample generated code
    sample_code = """
go atlas

# Define mesh
mesh space.mult=1.0
x.mesh loc=0.0 spac=0.05
y.mesh loc=0.0 spac=0.02

# Define regions
region num=1 material=Silicon

# Define electrodes
electrode name=source x.min=0.0 x.max=0.3
electrode name=gate x.min=0.3 x.max=0.7

# Doping
doping uniform conc=1e16 p.type

# Models
models cvt srh

# Solve
solve init
solve vgate=0.0 vstep=0.1 vfinal=3V

quit
"""
    
    # Expected parameters
    expected_params = ["L=1u", "W=10u", "VDD=3V", "material=Silicon"]
    
    # Reference code for BLEU
    reference_code = """
go atlas
mesh 
region material=Silicon
electrode name=source
electrode name=gate
doping conc=1e16
models srh
solve vgate=3V
quit
"""
    
    # Test all metrics
    results = metrics.evaluate_all(sample_code, expected_params, reference_code)
    
    # Display results
    print(f"SVS Score: {results['svs_score']:.3f}")
    print(f"  Sections found: {results['found_count']}/{results['total_required']}")
    print(f"  Missing: {results['sections_missing']}")
    
    print(f"\nPEM Score: {results['pem_score']:.3f}")
    print(f"  Matched: {results['matched_count']}/{results['total_expected']}")
    print(f"  Missing: {results['missing_params']}")
    
    print(f"\nCCS Score: {results['ccs_score']:.3f}")
    print(f"  Categories found: {results['categories_found']}/{results['total_categories']}")
    print(f"  Essential score: {results['essential_score']:.3f}")
    
    print(f"\nBLEU Score: {results['bleu_score']:.3f}")
    print(f"  BLEU-1: {results['bleu_1']:.3f}")
    
    print(f"\nComposite Score: {results['composite_score']:.3f}")
    
    print("\n✓ All metrics tested successfully!")


if __name__ == "__main__":
    test_metrics()