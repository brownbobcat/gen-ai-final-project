Task 1: Creating Folder Structure
- Creating the required directories for organizing the project components.

Set up Python virtual environment and install dependencies
  - Created Python virtual environment (.venv)
  - Installed all required packages including:
    - Deep learning: transformers, torch, peft, accelerate, bitsandbytes
    - RAG: sentence-transformers, faiss-cpu
    - Data handling: datasets, pandas, numpy
    - Evaluation: scikit-learn

Create preprocess.py - load JSON dataset and clean data
  1. Created a comprehensive preprocessing script that:
    - Loads the JSON dataset (713 examples)
    - Normalizes instruction text (removes extra whitespace, standardizes Unicode)
    - Cleans Silvaco code (maintains structure, standardizes formatting)
    - Creates appropriate prompts for the model
    - Tokenizes data with proper padding and label masking
    - Splits data into train (641) and validation (72) sets
  2. Key design decisions:
    - 90/10 train/validation split: Standard split for medium-sized datasets
    - 2048 max token length: Sufficient for most Silvaco code examples
    - Label masking: Only trains on the output code, not the prompt
    - Prompt template: Provides clear context and structure for generation
  3. Output files created:
    - processed_dataset/: HuggingFace Dataset format for easy training
    - dataset_info.json: Metadata about the processed dataset
    - processed_examples.json: Sample processed examples for verification

Create rag_retrieve.py - build embedding index with FAISS
- Building a Retrieval-Augmented Generation (RAG) system to find similar Silvaco examples during generation.

  Why this approach:
  1. Sentence-transformers: Using MiniLM model for efficient embeddings that capture semantic similarity
  2. FAISS index: Facebook's library for fast similarity search, perfect for finding relevant examples
  3. Embedding all .in files: Creates a comprehensive knowledge base from Silvaco examples
  4. Top-k retrieval: Retrieves the 3 most similar examples to provide context for generation

Create train_qlora.py - fine-tune Qwen model with QLoRA
- Building a training script to fine-tune the Qwen model using QLoRA (Quantized Low-Rank Adaptation).

  Why this approach:
  1. QLoRA: Enables fine-tuning large models on limited hardware by using 4-bit quantization
  2. LoRA parameters: r=16, alpha=32 provide good balance between performance and efficiency
  3. Target modules: q_proj, k_proj, v_proj are the attention projection layers most important for task-specific adaptation
  4. Training strategy: Small batch size (8) with gradient accumulation for stability
  5. Learning rate: 2e-4 is optimal for LoRA fine-tuning

Create generate.py - implement prompt + RAG + generation
- Building the generation script that combines the fine-tuned model (once trained) with RAG retrieval to generate Silvaco
  code.

  Why this approach:
  1. RAG Integration: Retrieves similar examples to provide context
  2. Flexible Model Loading: Can use either fine-tuned or base model
  3. Optimized Generation: Temperature and sampling parameters tuned for code generation
  4. Structured Prompting: Clear template that guides the model

Build benchmark test cases (20 custom test cases)
- Creating 20 custom test cases covering different semiconductor devices and edge cases for comprehensive evaluation.

  Why this approach:
  1. Device Coverage: MOSFETs (6), Diodes (3), BJTs (3), Photonic (2), Sensors (2), Edge cases (4)
  2. Comprehensive Testing: Each category tests different aspects of the model's knowledge
  3. Expected References: Include key parameters that should appear in generated code
  4. Progressive Complexity: From simple to complex device descriptions

Create metrics.py - implement evaluation metrics (SVS, PEM, CCS)
- Implementing the three required evaluation metrics: SVS, PEM, and CCS, plus optional BLEU similarity.

  Why this approach:
  1. Syntax Validity Score (SVS): Checks for required Silvaco sections and syntax
  2. Parameter Exact Match (PEM): Uses regex to extract and compare key parameters
  3. Component Completeness Score (CCS): Verifies presence of essential simulation components
  4. BLEU Score: Optional semantic similarity metric for overall quality

Create eval.py - full evaluation pipeline
- Building the complete evaluation script that runs all test cases, generates code, computes metrics, and saves results to
  CSV.

  Why this approach:
  1. Automated Pipeline: Runs all 20 test cases systematically
  2. Comprehensive Metrics: Computes SVS, PEM, CCS, and BLEU for each test
  3. CSV Output: Structured results for analysis and reporting
  4. Error Handling: Gracefully handles generation failures
  5. Progress Tracking: Shows progress during long evaluation runs
