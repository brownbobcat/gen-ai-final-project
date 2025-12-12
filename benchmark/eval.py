#!/usr/bin/env python3
"""
eval.py - Full evaluation pipeline for SPICE circuit code generation
Runs all test cases, generates code, computes metrics, saves results
"""

import os
import json
import csv
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import pandas as pd

# Add src to path for imports
import sys
sys.path.append('src')

from generate import SPICEModel
from metrics import SPICEMetrics


class SPICEEvaluator:
    """Complete evaluation pipeline for SPICE code generation"""
    
    def __init__(self, use_rag: bool = True, use_fine_tuned: bool = True):
        """Initialize evaluator"""
        self.use_rag = use_rag
        self.use_fine_tuned = use_fine_tuned
        
        print("Initializing SPICE Evaluation Pipeline")
        print("=" * 50)
        
        # Initialize generator
        print("Loading SPICE model...")
        base_model = "Qwen/Qwen2-0.5B"
        adapter_path = "model/adapter_model"
        self.generator = SPICEModel(base_model, adapter_path)
        
        # Note: RAG functionality would need to be integrated separately
        # For now, we'll use the model without RAG enhancement
        if use_rag:
            print("⚠️  RAG not integrated with SPICEModel yet - using model only")
            self.use_rag = False
        
        # Initialize metrics
        self.metrics = SPICEMetrics()
        print("✓ Metrics system loaded")
        
        # Results storage
        self.results = []
        
    def load_test_cases(self, filepath: str) -> List[Dict]:
        """Load test cases from JSON file"""
        print(f"Loading test cases from {filepath}")
        with open(filepath, 'r') as f:
            test_cases = json.load(f)
        print(f"✓ Loaded {len(test_cases)} test cases")
        return test_cases
    
    def evaluate_single_case(self, test_case: Dict, case_index: int) -> Dict:
        """Evaluate a single test case"""
        case_id = test_case['id']
        description = test_case['description']
        expected = test_case['expected']
        
        print(f"\nEvaluating case {case_index + 1}: {case_id}")
        print(f"Description: {description[:100]}...")
        
        result = {
            'case_id': case_id,
            'category': test_case['category'],
            'description': description,
            'generation_time': 0,
            'generation_success': False,
            'generated_code': '',
            'error_message': '',
            'svs_score': 0.0,
            'pem_score': 0.0,
            'ccs_score': 0.0,
            'composite_score': 0.0,
            'bleu_score': 0.0
        }
        
        try:
            # Generate code
            start_time = time.time()
            
            generated_code = self.generator.generate(
                description=description,
                max_new_tokens=600,
                temperature=0.7
            )
            
            generation_time = time.time() - start_time
            
            result.update({
                'generation_time': generation_time,
                'generation_success': True,
                'generated_code': generated_code
            })
            
            print(f"✓ Code generated ({generation_time:.1f}s)")
            
            # Evaluate with metrics
            expected_params = expected.get('parameters', [])
            
            metrics_result = self.metrics.evaluate_all(
                generated_code=generated_code,
                expected_params=expected_params,
                reference_code=None  # No reference code available
            )
            
            # Update result with metrics
            result.update({
                'svs_score': metrics_result['svs_score'],
                'pem_score': metrics_result['pem_score'],
                'ccs_score': metrics_result['ccs_score'],
                'composite_score': metrics_result['composite_score'],
                'bleu_score': metrics_result.get('bleu_score', 0.0),
                'svs_sections_found': len(metrics_result['sections_found']),
                'svs_sections_total': metrics_result['total_required'],
                'pem_matched_count': metrics_result['matched_count'],
                'pem_total_expected': metrics_result['total_expected'],
                'ccs_categories_found': metrics_result['categories_found'],
                'ccs_categories_total': metrics_result['total_categories']
            })
            
            print(f"✓ Metrics computed - Composite: {result['composite_score']:.3f}")
            
        except Exception as e:
            print(f"❌ Generation failed: {str(e)}")
            result['error_message'] = str(e)
        
        return result
    
    def run_full_evaluation(self, test_cases_path: str, output_dir: str, 
                           sample_size: Optional[int] = None) -> str:
        """Run complete evaluation pipeline"""
        # Load test cases
        test_cases = self.load_test_cases(test_cases_path)
        
        # Sample if requested
        if sample_size and sample_size < len(test_cases):
            import random
            test_cases = random.sample(test_cases, sample_size)
            print(f"Sampling {sample_size} test cases for evaluation")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Evaluate each case
        print(f"\nStarting evaluation of {len(test_cases)} test cases...")
        print("This may take some time depending on model and hardware")
        print("-" * 50)
        
        for i, test_case in enumerate(tqdm(test_cases, desc="Evaluating")):
            result = self.evaluate_single_case(test_case, i)
            self.results.append(result)
            
            # Save intermediate results every 5 cases
            if (i + 1) % 5 == 0:
                self._save_intermediate_results(output_dir, i + 1)
        
        # Save final results
        results_file = self._save_results(output_dir)
        
        # Generate summary
        self._generate_summary(output_dir)
        
        print(f"\n" + "=" * 50)
        print("Evaluation Complete!")
        print(f"Results saved to: {results_file}")
        
        return results_file
    
    def _save_intermediate_results(self, output_dir: str, count: int):
        """Save intermediate results"""
        temp_file = os.path.join(output_dir, f"temp_results_{count}.json")
        with open(temp_file, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def _save_results(self, output_dir: str) -> str:
        """Save final results to CSV and JSON"""
        timestamp = int(time.time())
        
        # Save detailed JSON
        json_file = os.path.join(output_dir, f"detailed_results_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save CSV for analysis
        csv_file = os.path.join(output_dir, "results.csv")
        
        # Prepare CSV data
        csv_data = []
        for result in self.results:
            csv_row = {
                'case_id': result['case_id'],
                'category': result['category'],
                'generation_success': result['generation_success'],
                'generation_time': f"{result['generation_time']:.2f}",
                'svs_score': f"{result['svs_score']:.3f}",
                'pem_score': f"{result['pem_score']:.3f}",
                'ccs_score': f"{result['ccs_score']:.3f}",
                'composite_score': f"{result['composite_score']:.3f}",
                'bleu_score': f"{result.get('bleu_score', 0):.3f}",
                'error_message': result.get('error_message', '')
            }
            csv_data.append(csv_row)
        
        # Write CSV
        with open(csv_file, 'w', newline='') as f:
            if csv_data:
                writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
                writer.writeheader()
                writer.writerows(csv_data)
        
        print(f"✓ Results saved to {csv_file}")
        return csv_file
    
    def _generate_summary(self, output_dir: str):
        """Generate evaluation summary"""
        if not self.results:
            return
        
        # Calculate overall statistics
        successful_cases = [r for r in self.results if r['generation_success']]
        failed_cases = [r for r in self.results if not r['generation_success']]
        
        summary = {
            'evaluation_info': {
                'total_cases': len(self.results),
                'successful_cases': len(successful_cases),
                'failed_cases': len(failed_cases),
                'success_rate': len(successful_cases) / len(self.results),
                'use_rag': self.use_rag,
                'use_fine_tuned': self.use_fine_tuned
            },
            'performance_metrics': {},
            'category_breakdown': {}
        }
        
        if successful_cases:
            # Overall metrics
            summary['performance_metrics'] = {
                'avg_svs_score': sum(r['svs_score'] for r in successful_cases) / len(successful_cases),
                'avg_pem_score': sum(r['pem_score'] for r in successful_cases) / len(successful_cases),
                'avg_ccs_score': sum(r['ccs_score'] for r in successful_cases) / len(successful_cases),
                'avg_composite_score': sum(r['composite_score'] for r in successful_cases) / len(successful_cases),
                'avg_generation_time': sum(r['generation_time'] for r in successful_cases) / len(successful_cases)
            }
            
            # Category breakdown
            from collections import defaultdict
            category_stats = defaultdict(list)
            
            for result in successful_cases:
                category_stats[result['category']].append(result)
            
            for category, results in category_stats.items():
                summary['category_breakdown'][category] = {
                    'count': len(results),
                    'avg_composite_score': sum(r['composite_score'] for r in results) / len(results),
                    'avg_svs_score': sum(r['svs_score'] for r in results) / len(results),
                    'avg_pem_score': sum(r['pem_score'] for r in results) / len(results),
                    'avg_ccs_score': sum(r['ccs_score'] for r in results) / len(results)
                }
        
        # Save summary
        summary_file = os.path.join(output_dir, "evaluation_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print(f"\n" + "=" * 50)
        print("EVALUATION SUMMARY")
        print("=" * 50)
        print(f"Total test cases: {summary['evaluation_info']['total_cases']}")
        print(f"Successful generations: {summary['evaluation_info']['successful_cases']}")
        print(f"Failed generations: {summary['evaluation_info']['failed_cases']}")
        print(f"Success rate: {summary['evaluation_info']['success_rate']:.1%}")
        
        if summary['performance_metrics']:
            print(f"\nAverage Scores:")
            print(f"  SVS (Syntax): {summary['performance_metrics']['avg_svs_score']:.3f}")
            print(f"  PEM (Parameters): {summary['performance_metrics']['avg_pem_score']:.3f}")
            print(f"  CCS (Completeness): {summary['performance_metrics']['avg_ccs_score']:.3f}")
            print(f"  Composite: {summary['performance_metrics']['avg_composite_score']:.3f}")
            print(f"  Avg generation time: {summary['performance_metrics']['avg_generation_time']:.1f}s")
        
        print(f"\n✓ Summary saved to {summary_file}")


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate SPICE code generation")
    parser.add_argument(
        "--test-cases",
        default="benchmark/test_cases.json", 
        help="Path to test cases JSON file"
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark/results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG retrieval"
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Evaluate only a sample of N test cases"
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = SPICEEvaluator(use_rag=not args.no_rag)
    
    # Run evaluation
    results_file = evaluator.run_full_evaluation(
        test_cases_path=args.test_cases,
        output_dir=args.output_dir,
        sample_size=args.sample
    )
    
    print(f"\nEvaluation pipeline complete!")
    print(f"Results available in: {args.output_dir}")
    print(f"Main results file: {results_file}")


if __name__ == "__main__":
    main()