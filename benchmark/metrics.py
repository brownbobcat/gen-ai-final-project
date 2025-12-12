#!/usr/bin/env python3
"""
metrics.py - Evaluation metrics for SPICE circuit code generation
Implements SVS, PEM, CCS, and BLEU metrics as required
"""

import re
import json
import math
from typing import Dict, List, Tuple, Optional
from collections import Counter
import numpy as np


class SPICEMetrics:
    """Evaluation metrics for SPICE circuit code generation"""
    
    def __init__(self):
        """Initialize metrics calculator"""
        # Required SPICE sections for syntax validation
        self.required_sections = [
            '.end',          # End of netlist (required)
            'analysis',      # Analysis command (.tran, .ac, .dc, .op)
            'components',    # Circuit components (M, R, C, L, V, I)
            'nodes'          # Node connections
        ]
        
        # Component categories for CCS
        self.component_categories = {
            'devices': ['m', 'q', 'd', 'j'],  # MOSFETs, BJTs, diodes, JFETs
            'passive': ['r', 'c', 'l'],       # Resistors, capacitors, inductors  
            'sources': ['v', 'i'],            # Voltage/current sources
            'analysis': ['.tran', '.ac', '.dc', '.op', '.noise', '.tf'],
            'directives': ['.param', '.model', '.include', '.lib', '.subckt'],
            'output': ['.print', '.probe', '.measure', '.save', '.plot']
        }
    
    def syntax_validity_score(self, generated_code: str) -> Dict[str, any]:
        """
        Syntax Validity Score (SVS)
        Detects presence of required SPICE sections
        Returns 1 if all required sections present, 0 otherwise
        """
        code_lower = generated_code.lower()
        
        # Check for each required section
        sections_found = []
        sections_missing = []
        
        # Check for .END statement (required)
        if '.end' in code_lower:
            sections_found.append('.end')
        else:
            sections_missing.append('.end')
        
        # Check for analysis commands
        analysis_commands = ['.tran', '.ac', '.dc', '.op', '.noise', '.tf']
        has_analysis = any(cmd in code_lower for cmd in analysis_commands)
        if has_analysis:
            sections_found.append('analysis')
        else:
            sections_missing.append('analysis')
        
        # Check for circuit components
        component_patterns = [r'\bm\d+', r'\br\d+', r'\bc\d+', r'\bl\d+', r'\bv\d+', r'\bi\d+', 
                             r'\bq\d+', r'\bd\d+', r'\bj\d+', r'\bx\d+']
        has_components = any(re.search(pattern, code_lower) for pattern in component_patterns)
        if has_components:
            sections_found.append('components')
        else:
            sections_missing.append('components')
        
        # Check for node connections (at least 2 nodes per component)
        node_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b|\b\d+\b'
        nodes = set(re.findall(node_pattern, code_lower))
        has_nodes = len(nodes) >= 2  # At least GND and one other node
        if has_nodes:
            sections_found.append('nodes')
        else:
            sections_missing.append('nodes')
        
        # Calculate score (binary: all or nothing for core sections)
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
        
        # Parameter extraction patterns for SPICE
        param_patterns = {
            'dimensions': r'(?:L|W|length|width)\s*=\s*([0-9.]+[unmpf]?)',
            'voltages': r'(?:V\w*|voltage|DC)\s*=?\s*([0-9.-]+[mV]?)',
            'resistance': r'(?:R|resistance)\s*=?\s*([0-9.e+-]+[kmg]?)',
            'capacitance': r'(?:C|capacitance)\s*=?\s*([0-9.e+-]+[unmpf]?[F]?)',
            'inductance': r'(?:L|inductance)\s*=?\s*([0-9.e+-]+[unm]?[H]?)',
            'frequency': r'(?:freq|frequency)\s*=?\s*([0-9.]+[GMK]?[Hh]z)',
            'current': r'(?:I\w*|current|DC)\s*=?\s*([0-9.e+-]+[unm]?[A]?)',
            'temperature': r'(?:temp|temperature)\s*=?\s*([0-9.-]+)',
            'models': r'(?:\.model|model)\s+([A-Za-z0-9\s_]+)',
            'subckt': r'(?:\.subckt)\s+([A-Za-z0-9\s_]+)'
        }
        
        # Extract parameters from generated code
        extracted_params = []
        code_lower = generated_code.lower()
        
        for param_type, pattern in param_patterns.items():
            matches = re.findall(pattern, code_lower, re.IGNORECASE)
            for match in matches:
                extracted_params.append(f"{param_type}={match}")
        
        # Also look for direct SPICE parameter matches
        direct_patterns = [
            r'([LW])\s*=\s*([0-9.]+[unmpf]?)',        # L=1u, W=10u
            r'(VDD|VCC|Vdd|DC)\s*=?\s*([0-9.-]+[mV]?)', # VDD=3V, DC 5V
            r'([RC])\s*=?\s*([0-9.e+-]+[kmgpnuf]?)',  # R=1k, C=10p
            r'(\w+)\s*=\s*([0-9.e+-]+[unmpf]?[VHA]?)', # Generic param=value
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
        Check presence of essential SPICE simulation components
        """
        code_lower = generated_code.lower()
        
        # Check each component category
        category_scores = {}
        found_components = {}
        
        for category, keywords in self.component_categories.items():
            found_in_category = []
            
            for keyword in keywords:
                if category in ['devices', 'passive', 'sources']:
                    # For components, check for patterns like M1, R1, etc.
                    pattern = rf'\b{keyword}\d+'
                    if re.search(pattern, code_lower):
                        found_in_category.append(keyword)
                else:
                    # For directives and analysis, check direct keyword match
                    if keyword in code_lower:
                        found_in_category.append(keyword)
            
            # Score for this category (binary: at least one keyword found)
            category_scores[category] = 1.0 if found_in_category else 0.0
            found_components[category] = found_in_category
        
        # Overall CCS score (average of category scores)
        ccs_score = sum(category_scores.values()) / len(category_scores)
        
        # Count essential components for SPICE
        essential_components = ['analysis', 'sources']  # Must have analysis and sources
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
    print("Testing SPICE Evaluation Metrics")
    print("=" * 50)
    
    metrics = SPICEMetrics()
    
    # Sample generated SPICE code
    sample_code = """
* Simple NMOS Amplifier Circuit
.subckt nmos_amp in out vdd gnd
M1 out in gnd gnd nmos W=10u L=1u
R1 vdd out 1k
C1 in ac_in 1p
.ends

* Main circuit
VDD vdd 0 DC 5V
Vin ac_in 0 DC 2.5V AC 1mV
X1 ac_in output vdd 0 nmos_amp
RL output 0 10k

* Analysis
.model nmos nmos level=1 vto=1 kp=50u
.op
.ac dec 10 1 100meg
.tran 1n 100n

.end
"""
    
    # Expected parameters for SPICE
    expected_params = ["W=10u", "L=1u", "DC=5V", "R=1k"]
    
    # Reference code for BLEU
    reference_code = """
* NMOS Circuit
M1 out in gnd gnd nmos W=10u L=1u
VDD vdd 0 DC 5V
R1 vdd out 1k
.model nmos nmos level=1
.op
.end
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