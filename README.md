# Silvaco TCAD Code Generation with Fine-Tuned LLM

A specialized language model system for generating Silvaco ATLAS simulation code from natural language descriptions. This project combines domain-specific fine-tuning with Retrieval-Augmented Generation (RAG) to automate semiconductor device simulation code creation.

## 🎯 Overview

This system addresses the challenge of converting natural language device descriptions into precise Silvaco ATLAS simulation decks. It features:

- **Fine-tuned Qwen2-0.5B** model with LoRA adapters (4.4M trainable parameters, 0.88% of total model)
- **Enhanced Few-Shot Learning** with professional TCAD templates and pattern analysis
- **RAG-enhanced generation** using vector search over simulation databases  
- **Comprehensive benchmark** with 20 diverse test cases across device categories
- **Multi-metric evaluation** framework with domain-specific metrics
- **Assignment 4 Integration**: Advanced prompt engineering with 152% performance improvement

## 📁 Project Structure

```
project solution/
├── src/                          # Source code
│   ├── generate.py              # Main generation script (with Few-Shot Learning)
│   ├── enhanced_prompts.py      # Professional TCAD prompt templates
│   ├── prompt_engineering.py    # Assignment 4: Prompt engineering techniques  
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
├── prompt_engineering_report.md # Assignment 4: Prompt engineering report
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

**Enhanced System with Few-Shot Learning:**
- **SVS**: 0.75 (excellent syntax structure with Few-Shot templates)
- **PEM**: 0.14 (parameter extraction improved but challenging)  
- **CCS**: 0.90 (excellent component coverage)
- **Composite**: 0.59 (152% improvement over baseline - Assignment 4 results)

**Original System Performance:**
- **SVS**: 0.50 (baseline syntax structure learning)
- **PEM**: 0.17 (baseline parameter extraction)  
- **CCS**: 0.58 (baseline component coverage)
- **Composite**: 0.42 (baseline performance)

### Training Performance

**Training Efficiency (LoRA vs Full Fine-tuning):**
- **Parameter efficiency**: 0.88% trainable parameters (4.4M vs 498M total)
- **Memory efficiency**: Significant VRAM reduction during training
- **Training speed**: 3.47 samples/second (88.8 seconds for 78 steps)
- **Convergence**: 2 epochs sufficient (training loss: 2.389 → 1.987 validation loss)

**Loss Progression:**
```
Epoch 1: Training Loss 2.811 → Validation Loss 2.212  
Epoch 2: Training Loss 2.146 → Validation Loss 1.987
```

**Hardware Requirements:**
- **Training**: GPU recommended (bfloat16 mixed precision)
- **Inference**: CPU compatible (FP32 fallback for Mac M-series)
- **Memory**: Gradient checkpointing + LoRA enables training on modest hardware

## 🛠️ Development Guide

### Training Your Own Model

#### Current Model Configuration

**LoRA Parameters:**
- **Rank (r)**: 8
- **Alpha**: 16  
- **Dropout**: 0.05
- **Target modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Trainable parameters**: 4,399,104 (0.88% of 498,431,872 total parameters)

**Training Configuration:**
- **Batch size**: 1 (with gradient accumulation steps: 4)
- **Learning rate**: 2e-4
- **Epochs**: 2
- **Max length**: 384 tokens
- **Mixed precision**: bfloat16
- **Gradient checkpointing**: Enabled

**Training Results:**
- **Final training loss**: 2.389
- **Final validation loss**: 1.987
- **Training time**: 88.8 seconds (78 steps)
- **Training speed**: 3.47 samples/sec

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

#### LoRA Configuration Example

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,                  
    lora_alpha=16,        
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
)

model = get_peft_model(model, lora_config)
# trainable params: 4,399,104 || all params: 498,431,872 || trainable%: 0.8826
```

#### Tokenization Configuration

```python
from functools import partial

def tokenize(example, tokenizer):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=384
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenize_fn = partial(tokenize, tokenizer=tokenizer)
train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(tokenize_fn, batched=True, remove_columns=eval_dataset.column_names)
```

#### Training Arguments

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./spice-lora-qwen2",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=2e-4,
    logging_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    bf16=True,
    fp16=False,
    gradient_checkpointing=True,
    report_to="none",
    label_names=["labels"],
)
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

### Model Architecture

**Base Model**: Qwen2-0.5B (498,431,872 parameters)
- **Architecture**: Transformer decoder with RMSNorm and SwiGLU activation
- **Vocabulary size**: ~152K tokens
- **Context length**: 32,768 tokens (training truncated to 384)
- **Attention**: Multi-head attention with RoPE (Rotary Position Embedding)

**LoRA Adaptation**: 
- **Efficiency**: Only 0.88% of parameters are trainable (4.4M out of 498M)
- **Memory**: Significantly reduces VRAM requirements during training
- **Target layers**: All attention projection layers + MLP projections
- **Adaptation rank**: 8 (balance between capacity and efficiency)

### Generation Parameters

Edit generation parameters in `src/generate.py`:
- `temperature`: Randomness (0.1 = deterministic, 1.0 = creative)
- `top_p`: Nucleus sampling threshold  
- `repetition_penalty`: Reduce repetitive output
- `max_new_tokens`: Maximum generated length

**Current defaults**:
```python
outputs = model.generate(
    **inputs,
    max_new_tokens=600,
    temperature=0.2,
    do_sample=True,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id,
)
```

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


