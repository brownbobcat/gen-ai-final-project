#!/usr/bin/env python3
"""
prompt_experiments.py - Experimental framework for Assignment 4
Tests 4 prompt engineering techniques against baseline
"""

import os
import json
import time
from typing import Dict, List, Tuple

try:
    import pandas as pd
except ImportError:
    print("pandas not available, will skip CSV export")
    pd = None

# Import existing components
from generate import SPICEModel
from prompt_engineering import PromptEngineeringTechniques
import sys
import os

# Add benchmark directory to path
benchmark_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'benchmark')
sys.path.append(benchmark_path)
from metrics import SPICEMetrics


class PromptExperimentFramework:
    """Framework for testing prompt engineering techniques"""
    
    def __init__(self):
        """Initialize experimental framework"""
        # Get correct adapter path relative to project root
        project_root = os.path.dirname(os.path.dirname(__file__))
        adapter_path = os.path.join(project_root, "model", "adapter_model")
        base_model = "Qwen/Qwen2-0.5B"
        
        self.generator = SPICEModel(base_model, adapter_path)
        self.metrics = SPICEMetrics()
        self.prompt_techniques = PromptEngineeringTechniques()
        self.techniques = ['baseline', 'cot', 'few_shot', 'decomposition', 'format_control']
        
        # Select test cases for experiments (subset of benchmark)
        self.test_cases = self._select_test_cases()
        
        # Results storage
        self.results = {technique: [] for technique in self.techniques}
        
    def _get_technique_prompt(self, technique: str, description: str) -> str:
        """Get prompt for specific technique"""
        if technique == 'baseline':
            return self.prompt_techniques.baseline_prompt(description)
        elif technique == 'cot':
            return self.prompt_techniques.chain_of_thought_prompt(description)
        elif technique == 'few_shot':
            return self.prompt_techniques.few_shot_learning_prompt(description)
        elif technique == 'decomposition':
            return self.prompt_techniques.problem_decomposition_prompt(description)
        elif technique == 'format_control':
            return self.prompt_techniques.output_format_control_prompt(description)
        else:
            raise ValueError(f"Unknown technique: {technique}")
        
    def _select_test_cases(self) -> List[Dict]:
        """Select 8 representative test cases from benchmark for experiments"""
        # Load full test cases
        project_root = os.path.dirname(os.path.dirname(__file__))
        test_cases_path = os.path.join(project_root, "benchmark", "test_cases.json")
        with open(test_cases_path, 'r') as f:
            all_cases = json.load(f)
        
        # Select diverse subset covering different device categories
        selected_cases = [
            # Basic MOSFET
            next(case for case in all_cases if case['id'] == 'mosfet_01'),
            # Advanced MOSFET  
            next(case for case in all_cases if case['id'] == 'mosfet_03'),
            # Basic Diode
            next(case for case in all_cases if case['id'] == 'diode_01'),
            # BJT
            next(case for case in all_cases if case['id'] == 'bjt_01'),
            # Photonic device
            next(case for case in all_cases if case['id'] == 'photonic_01'),
            # Sensor device
            next(case for case in all_cases if case['id'] == 'sensor_02'),
            # Power device
            next(case for case in all_cases if case['id'] == 'mosfet_04'),
            # Edge case
            next(case for case in all_cases if case['id'] == 'edge_01')
        ]
        
        return selected_cases
    
    def run_single_experiment(self, technique: str, test_case: Dict) -> Dict:
        """Run single experiment with specified technique and test case"""
        print(f"  Running {technique} on {test_case['id']}...")
        
        description = test_case['description']
        expected = test_case['expected']
        
        # Generate prompt using technique
        start_time = time.time()
        
        try:
            # Get technique-specific prompt
            prompt = self._get_technique_prompt(technique, description)
            
            # Generate code using the model 
            generated_code = self.generator.generate(
                prompt,
                max_new_tokens=600,
                temperature=0.1
            )
            
            generation_time = time.time() - start_time
            generation_success = True
            error_message = ""
            
        except Exception as e:
            generation_time = time.time() - start_time
            generation_success = False
            generated_code = ""
            error_message = str(e)
            print(f"    Error: {error_message}")
        
        # Evaluate generated code
        if generation_success:
            # Calculate metrics
            svs_result = self.metrics.syntax_validity_score(generated_code)
            pem_result = self.metrics.parameter_exact_match(
                generated_code, expected.get('parameters', [])
            )
            ccs_result = self.metrics.component_completeness_score(generated_code)
            
            # Extract scores
            svs_score = svs_result['svs_score']
            pem_score = pem_result['pem_score']
            ccs_score = ccs_result['ccs_score']
            composite_score = (svs_score + pem_score + ccs_score) / 3
            
        else:
            svs_score = pem_score = ccs_score = composite_score = 0.0
        
        # Store result
        result = {
            'case_id': test_case['id'],
            'category': test_case['category'],
            'technique': technique,
            'generation_success': generation_success,
            'generation_time': generation_time,
            'svs_score': svs_score,
            'pem_score': pem_score,
            'ccs_score': ccs_score,
            'composite_score': composite_score,
            'error_message': error_message,
            'generated_code_length': len(generated_code),
            'prompt_length': len(prompt)
        }
        
        return result
    
    def run_all_experiments(self) -> None:
        """Run complete experimental suite"""
        print("=== PROMPT ENGINEERING EXPERIMENTS ===")
        print(f"Testing {len(self.techniques)} techniques on {len(self.test_cases)} test cases")
        print(f"Techniques: {', '.join(self.techniques)}")
        print(f"Test cases: {[case['id'] for case in self.test_cases]}")
        print()
        
        total_experiments = len(self.techniques) * len(self.test_cases)
        experiment_count = 0
        
        for technique in self.techniques:
            print(f"--- TECHNIQUE: {technique.upper()} ---")
            
            for test_case in self.test_cases:
                experiment_count += 1
                print(f"[{experiment_count}/{total_experiments}] ", end="")
                
                result = self.run_single_experiment(technique, test_case)
                self.results[technique].append(result)
                
                print(f"    Composite: {result['composite_score']:.3f}")
            
            print()
        
        print("All experiments completed!")
    
    def save_results(self, output_dir: str = "prompt_experiments") -> None:
        """Save experimental results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save raw results
        with open(f"{output_dir}/raw_results.json", 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Create comparison table
        all_results = []
        for technique, results in self.results.items():
            all_results.extend(results)
        
        if pd is not None:
            df = pd.DataFrame(all_results)
            df.to_csv(f"{output_dir}/comparison_results.csv", index=False)
        else:
            print("Skipping CSV export (pandas not available)")
        
        print(f"Results saved to {output_dir}/")
    
    def analyze_results(self) -> Dict:
        """Analyze experimental results"""
        print("\n=== RESULTS ANALYSIS ===")
        
        analysis = {}
        
        # Calculate averages per technique
        for technique in self.techniques:
            results = self.results[technique]
            
            avg_composite = sum(r['composite_score'] for r in results) / len(results)
            avg_svs = sum(r['svs_score'] for r in results) / len(results)
            avg_pem = sum(r['pem_score'] for r in results) / len(results)
            avg_ccs = sum(r['ccs_score'] for r in results) / len(results)
            avg_time = sum(r['generation_time'] for r in results) / len(results)
            success_rate = sum(r['generation_success'] for r in results) / len(results)
            
            analysis[technique] = {
                'avg_composite_score': avg_composite,
                'avg_svs_score': avg_svs,
                'avg_pem_score': avg_pem,
                'avg_ccs_score': avg_ccs,
                'avg_generation_time': avg_time,
                'success_rate': success_rate,
                'total_cases': len(results)
            }
            
            print(f"{technique.upper():15} | "
                  f"Composite: {avg_composite:.3f} | "
                  f"SVS: {avg_svs:.3f} | "
                  f"PEM: {avg_pem:.3f} | "
                  f"CCS: {avg_ccs:.3f} | "
                  f"Time: {avg_time:.1f}s")
        
        # Find best technique
        best_technique = max(analysis.keys(), 
                           key=lambda t: analysis[t]['avg_composite_score'])
        
        print(f"\nBest performing technique: {best_technique.upper()}")
        print(f"Best composite score: {analysis[best_technique]['avg_composite_score']:.3f}")
        
        return analysis
    
    def create_example_outputs(self, output_dir: str = "prompt_experiments") -> None:
        """Create example outputs for report"""
        print("\n=== CREATING EXAMPLE OUTPUTS ===")
        
        # Select one representative case for detailed examples
        example_case = next(case for case in self.test_cases if case['id'] == 'mosfet_01')
        
        examples = {}
        
        for technique in self.techniques:
            print(f"Generating example for {technique}...")
            
            try:
                prompt = self._get_technique_prompt(technique, example_case['description'])
                generated_code = self.generator.generate(prompt, max_new_tokens=600, temperature=0.1)
                
                examples[technique] = {
                    'prompt': prompt,
                    'generated_code': generated_code,
                    'case': example_case
                }
                
            except Exception as e:
                print(f"  Error generating example for {technique}: {e}")
                examples[technique] = {
                    'prompt': f"Error generating prompt: {e}",
                    'generated_code': f"Error generating code: {e}",
                    'case': example_case
                }
        
        # Save examples
        with open(f"{output_dir}/example_outputs.json", 'w') as f:
            json.dump(examples, f, indent=2)
        
        print(f"Example outputs saved to {output_dir}/example_outputs.json")


def main():
    """Main experimental runner"""
    print("Assignment 4: Prompt Engineering Experiments")
    print("=" * 50)
    
    # Initialize framework
    framework = PromptExperimentFramework()
    
    # Run all experiments
    framework.run_all_experiments()
    
    # Analyze results
    analysis = framework.analyze_results()
    
    # Save everything
    framework.save_results()
    framework.create_example_outputs()
    
    print("\nExperiment completed! Check prompt_experiments/ for results.")


if __name__ == "__main__":
    main()