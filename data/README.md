# Silvaco Device Simulation Code Dataset - Training Set

## Overview

This dataset contains **713 instruction-code pairs** for training Large Language Models to generate Silvaco device simulation code from natural language circuit design specifications.

**Task**: Given a detailed circuit design description (instruction), generate complete Silvaco SPICE netlist code (output).

**Important**: This is the **training set only**. An additional test set of 97 samples is reserved by the instructor for final evaluation.

## Dataset Files

### 1. `silvaco_dataset_train.json` (5.9MB)

Training dataset with 713 samples in instruction-following format.

**Structure**:
```json
[
  {
    "instruction": "Detailed circuit design specification...",
    "input": "",
    "output": "Complete Silvaco SPICE netlist code..."
  },
  "..."
]
```

**Fields**:
- `instruction`: Natural language description of circuit design requirements (avg 1,156 characters)
- `input`: Always empty (not used)
- `output`: Generated Silvaco SPICE netlist code (avg 7,393 characters)

**Statistics**:
- Total samples: 713
- Average instruction length: ~290 words
- Average code length: ~1,850 words
- Code types: SPICE netlists, harmonic balance simulations, transient analysis

### 2. `Silvaco_Examples_Student.zip` (18MB)

Reference examples from Silvaco user manuals (optional resource).

**Contents**:
- 726 `.in` files: Silvaco input deck files
- 76 `.lib` files: Model library files
- PDF manuals: Reference documentation

**Purpose**:
- Additional context and examples for understanding Silvaco syntax and circuit patterns
- **You can use these .in files to create your own training data** by writing corresponding instruction descriptions
- Provides opportunity for data augmentation and creative exploration

## Dataset Split Recommendation

Since you have 713 training samples, recommended split:

```
Training set:   570 samples (80%)
Validation set:  143 samples (20%)
```

Or if you prefer a validation + test split for local evaluation:

```
Training set:   570 samples (80%)
Validation set:  71 samples (10%)
Local test set:  72 samples (10%)
```

**Note**: Your model will be evaluated on the instructor's hidden test set (97 samples), not your local test set.

## Data Augmentation Opportunities

### Using Reference .in Files

The provided `Silvaco_Examples_Student.zip` contains 726 additional `.in` files. You can:

1. **Analyze the circuit code** in these `.in` files
2. **Write your own instruction descriptions** explaining what the circuit does
3. **Expand your training dataset** beyond the provided 713 samples
4. **Practice understanding** Silvaco code structure and conventions

This allows you to exercise creativity while expanding your training data if desired.

## Circuit Types in Dataset

The dataset covers various RF and mixed-signal circuit designs:

- **Voltage-Controlled Oscillators (VCO)**: Phase noise, harmonic balance analysis
- **Low-Noise Amplifiers (LNA)**: S-parameters, stability, IIP3 characterization
- **Mixers**: Conversion gain, noise figure, two-tone analysis
- **Switched-Capacitor Circuits**: Transient and periodic steady-state analysis
- **Operational Amplifiers**: DC/AC analysis, stability margins
- **Power Amplifiers**: Load-pull, compression point, linearity
- **TCAD Simulations**: Process, mesh, and device simulations

## Code Format

All generated code follows Silvaco SPICE netlist format:

```spice
* Circuit Name
* Netlist Generator Information

* Schematic name: CircuitName
[Component definitions: R, C, L, M, I, V]
[Subcircuit definitions: .SUBCKT ... .ENDS]

.GLOBAL [global nodes]

* Simulation Control
.inc [model files]
.OP
.PARAM [parameters]
.OPTIONS [simulation options]
.TEMP [temperature]

* Analysis Commands
.AC / .TRAN / .HARMONIC / .HNET / .HNOISE / etc.

.END
```

## Evaluation Metrics

Your model will be evaluated on the hidden test set using these metrics:

1. **Syntax Correctness (25%)**: Can code be parsed as valid SPICE?
   - Valid SPICE syntax
   - Proper command structure
   - No syntax errors

2. **Semantic Accuracy (40%)**: Is circuit topology correct?
   - Correct component connections
   - Accurate parameter values
   - Proper analysis directives

3. **Completeness (20%)**: Is all required information present?
   - All components defined
   - Simulation commands included
   - Proper termination (.END)

4. **Code Quality (15%)**:
   - Proper formatting and indentation
   - Meaningful node names
   - Complete netlist structure

## Data Loading Example

```python
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from functools import partial

# Load training dataset
with open('silvaco_dataset_train.json', 'r') as f:
    data = json.load(f)

# Load Qwen2 tokenizer (as implemented)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
# Tokenization function (as implemented)
def tokenize(example, tokenizer):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=384                    # Truncated for efficiency
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

# Apply tokenization
tokenize_fn = partial(tokenize, tokenizer=tokenizer)
train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=train_dataset.column_names)

# Tokenizer configuration notes:
# - PAD/BOS/EOS tokens aligned: pad_token_id: 151643, bos_token_id: None
# - use_cache=True incompatible with gradient checkpointing (automatically disabled)
```

## Important Notes

### Current Model Implementation

**Model Used**: Qwen2-0.5B (498,431,872 parameters)
- **Training method**: LoRA (Low-Rank Adaptation) 
- **Trainable parameters**: 4,399,104 (0.88% of total)
- **Context length**: 384 tokens (truncated for efficiency)
- **Mixed precision**: bfloat16 for stable training

**Actual Training Configuration**:
```python
# LoRA Configuration
lora_config = LoraConfig(
    r=8,                    # Rank
    lora_alpha=16,          # Scaling factor  
    lora_dropout=0.05,      # Dropout rate
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                   "gate_proj", "up_proj", "down_proj"]
)

# Training Arguments  
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,     # Effective batch size: 4
    num_train_epochs=2,
    learning_rate=2e-4,
    max_length=384,                    # Truncated context
    bf16=True,                         # Mixed precision
    gradient_checkpointing=True        # Memory optimization
)
```

**Training Results (2 epochs, 78 steps)**:
- **Training time**: 88.8 seconds
- **Final training loss**: 2.389  
- **Final validation loss**: 1.987
- **Training speed**: 3.47 samples/second
- **Memory efficiency**: ~0.88% trainable parameters vs full fine-tuning

### Context Length Strategy
- **Original samples**: ~8,500 characters (~2,100 tokens) 
- **Training truncation**: 384 tokens (for efficiency and memory)
- **Generation context**: Up to 32,768 tokens (full model capacity)
- **Strategy**: Aggressive truncation during training, full context during inference

### Training Recommendations (Implemented)
- **Fine-tuning method**: LoRA (implemented)
- **Batch size**: 1 + gradient accumulation (implemented)
- **Learning rate**: 2e-4 (implemented)
- **Epochs**: 2 epochs (implemented)  
- **Validation**: Epoch-based evaluation (implemented)

### Hidden Test Set
- An additional 97 samples are reserved for instructor evaluation
- These samples are NOT in your training data
- Your model will be evaluated on these unseen samples
- Focus on generalization, not memorization

## Data Quality Notes

- All code samples are from Silvaco user manuals and verified examples
- Instructions are synthetically generated to describe the circuits
- Code follows proper SPICE syntax and simulation best practices
- Some samples may reference external library files (.inc directives)

## License and Usage

- **For educational purposes only**
- Source: Silvaco user manuals and documentation
- Use for course final project
- Do not redistribute without permission

## Questions?

- See `final_project_instructions.html` for complete project details
- Ask in office hours or Slack channel #final-project
- Check Silvaco documentation in `Silvaco_Examples_Student.zip`

---

**Dataset prepared for CSC 375/575 - Generative AI Final Project**
**Prof. Rongyu Lin, Quinnipiac University, Fall 2025**
