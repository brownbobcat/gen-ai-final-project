# Silvaco TCAD Code Generation with Fine-Tuned LLM

A specialized language model system for generating Silvaco ATLAS simulation code from natural language descriptions. This project combines domain-specific fine-tuning with Retrieval-Augmented Generation (RAG) to automate semiconductor device simulation code creation.

## 🎯 Overview

This system addresses the challenge of converting natural language device descriptions into precise Silvaco ATLAS simulation decks. It features:

- **Fine-tuned Qwen2-0.5B** model with LoRA adapters trained on 726 TCAD examples
- **RAG-enhanced generation** using vector search over simulation databases
- **Comprehensive benchmark** with 20 diverse test cases across device categories
- **Multi-metric evaluation** framework with domain-specific metrics

## 📁 Project Structure

```
project solution/
├── src/                          # Source code
│   ├── generate.py              # Main generation script
│   ├── train_qlora.py           # Training script (for GPU)
│   ├── train_qlora_cpu.py       # CPU training demo
│   ├── preprocess.py            # Data preprocessing
│   └── rag_retrieve.py          # RAG implementation
├── model/                        # Model files
│   ├── adapter_model/           # LoRA adapters
│   └── tokenizer/               # Tokenizer files
├── data/                        # Training data
│   ├── raw/                     # Original Silvaco files
│   └── processed/               # Preprocessed datasets
├── benchmark/                    # Evaluation framework
│   ├── test_cases.json          # 20 test cases
│   ├── eval.py                  # Evaluation pipeline
│   ├── metrics.py               # Custom metrics
│   ├── benchmark_design.md      # Design documentation
│   └── test_results/            # Evaluation results
├── technical_report.md          # Technical report (4 pages)
└── README.md                    # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- 16GB+ RAM recommended
- Virtual environment recommended

### Installation

1. **Clone and setup**:
```bash
git clone <repository>
cd "project solution"
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify model files**:
```bash
ls model/adapter_model/  # Should show adapter_config.json and adapter_model.safetensors
ls model/tokenizer/      # Should show tokenizer files
```

### Basic Usage

#### Generate Silvaco Code

```bash
# Basic generation
python src/generate.py \
  --input "Design a basic NMOS transistor for digital logic" \
  --output result.in

# With custom parameters
python src/generate.py \
  --input "Create a VCO using NMOS cross-coupled pair with LC tank" \
  --adapter_path model/adapter_model \
  --output vco_simulation.in \
  --temperature 0.1 \
  --max-length 1024
```

#### Run Benchmark Evaluation

```bash
# Quick test (2 cases)
python benchmark/eval.py --sample 2 --output-dir benchmark/results

# Full evaluation (20 cases) 
python benchmark/eval.py --output-dir benchmark/results

# No RAG (faster)
python benchmark/eval.py --no-rag --sample 5 --output-dir benchmark/results
```

## 📊 Evaluation Framework

### Metrics Implemented

1. **Syntax Validity Score (SVS)**
   - Checks for required Silvaco sections (`go atlas`, `mesh`, `region`, etc.)
   - Binary score: 1.0 if all present, 0.0 otherwise

2. **Parameter Exact Match (PEM)**
   - Extracts and matches device parameters (L, W, voltages, doping)
   - Score: percentage of expected parameters found

3. **Component Completeness Score (CCS)**  
   - Evaluates presence of simulation components across 6 categories
   - Score: average of category completion rates

4. **BLEU Similarity (Optional)**
   - Textual similarity when reference code available
   - 4-gram BLEU with brevity penalty

### Test Cases

20 diverse test cases covering:
- **Basic devices** (5): NMOS, PMOS, diodes, BJTs, JFETs
- **Advanced devices** (3): FinFETs, SOI devices, tunnel FETs
- **Analog circuits** (4): Op-amps, current mirrors, differential pairs  
- **RF devices** (3): VCOs, LNAs, RF MOSFETs
- **Power/sensors** (4): Power devices, MEMS, photonics
- **Edge cases** (1): Conflicting specs, missing parameters

### Performance Baselines

Current model performance on sample evaluation:
- **SVS**: 0.50 (syntax structure learning)
- **PEM**: 0.17 (parameter extraction challenging)  
- **CCS**: 0.58 (good component coverage)
- **Composite**: 0.42 (moderate overall performance)

## 🛠️ Development Guide

### Training Your Own Model

1. **Prepare dataset**:
```bash
python src/preprocess.py --input data/raw --output data/processed
```

2. **Train on GPU** (recommended):
```bash
# Transfer to GPU system and run
python src/train_qlora.py
```

3. **CPU demo** (for development):
```bash
python src/train_qlora_cpu.py  # Creates mock adapters
```

### Extending the Benchmark

1. **Add test cases** to `benchmark/test_cases.json`:
```json
{
  "id": "new_case_01",
  "category": "Custom",
  "description": "Your device description here",
  "expected": {
    "device_type": "device_name",
    "parameters": ["L=1u", "W=10u"],
    "analysis": ["DC", "solve"]
  }
}
```

2. **Run validation**:
```bash
python benchmark/validate_testcases.py
```

### Custom Metrics

Extend `benchmark/metrics.py` to add domain-specific evaluations:

```python
class SilvacoMetrics:
    def custom_metric(self, generated_code, expected_features):
        # Your metric implementation
        return {"custom_score": score, "details": analysis}
```

## 📋 Available Commands

### Generation Options

```bash
python src/generate.py [OPTIONS]
  --input, -i         Input description (required)
  --output, -o        Output file (default: output.in)
  --adapter_path      Path to LoRA adapters
  --temperature       Generation temperature (default: 0.7)
  --max-length        Max sequence length (default: 2048)
  --no-rag           Disable RAG retrieval
```

### Evaluation Options

```bash
python benchmark/eval.py [OPTIONS]
  --test-cases       Path to test cases JSON
  --output-dir       Results directory
  --sample N         Evaluate only N random cases
  --no-rag          Disable RAG for faster evaluation
```

### Metric Testing

```bash
python benchmark/metrics.py  # Test metrics with sample data
```

## 🔧 Configuration

### Model Parameters

Edit generation parameters in `src/generate.py`:
- `temperature`: Randomness (0.1 = deterministic, 1.0 = creative)
- `top_p`: Nucleus sampling threshold  
- `repetition_penalty`: Reduce repetitive output
- `max_new_tokens`: Maximum generated length

### RAG Settings

Configure retrieval in `src/rag_retrieve.py`:
- `embedding_model`: Sentence transformer model
- `k_examples`: Number of retrieved examples
- `index_type`: FAISS index configuration

## 🐛 Troubleshooting

### Common Issues

1. **ModuleNotFoundError: torch**
   ```bash
   source .venv/bin/activate  # Activate virtual environment
   pip install torch transformers
   ```

2. **CUDA/GPU errors**
   ```bash
   # Force CPU usage
   export CUDA_VISIBLE_DEVICES=""
   ```

3. **Memory errors during evaluation**
   ```bash
   # Use smaller sample size
   python benchmark/eval.py --sample 5 --no-rag
   ```

4. **RAG index not found**
   ```bash
   # Disable RAG if embeddings not available
   python src/generate.py --no-rag --input "your description"
   ```

### Performance Optimization

- **CPU inference**: Model runs on CPU by default (slower but compatible)
- **Batch evaluation**: Use `--sample N` for quick testing
- **Disable RAG**: Use `--no-rag` for faster generation
- **Lower precision**: Model uses FP32 for Mac compatibility

## 📈 Results Interpretation

### Evaluation Outputs

Results are saved in multiple formats:
- `results.csv`: Summary scores for analysis
- `detailed_results_*.json`: Complete evaluation data
- `evaluation_summary.json`: Statistical overview

### Score Interpretation

- **SVS = 1.0**: Perfect syntax structure
- **PEM > 0.5**: Good parameter extraction  
- **CCS > 0.7**: Comprehensive simulation coverage
- **Composite > 0.6**: Strong overall performance

### Category Analysis

Check category breakdown in results for:
- Device-specific performance patterns
- Complexity-dependent score distributions  
- Failure mode identification

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-metric`)
3. Add tests for new functionality
4. Submit pull request with detailed description

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black src/ benchmark/
```

## 📚 References

- [Silvaco ATLAS Documentation](https://silvaco.com/products/tcad/device_simulation/atlas/)
- [Qwen2 Model Documentation](https://huggingface.co/Qwen/Qwen2-0.5B)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [RAG: Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)

## 📄 License

This project is for academic use. See individual component licenses for specific terms.

## 👥 Authors

- Nicholas Brown
- Nirmit Dagli
- Tamuka Manjemu
- Project for Generative AI Course, Fall 2025

## 🙏 Acknowledgments

- Silvaco for TCAD simulation examples
- Hugging Face for model infrastructure
- Course instructors and TAs for guidance


