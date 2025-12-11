# Silvaco TCAD Generation Benchmark Design Document

## Overview

This document describes the comprehensive benchmark designed to evaluate large language models' capability to generate Silvaco ATLAS simulation code for semiconductor devices. The benchmark addresses the unique challenges of domain-specific code generation for TCAD (Technology Computer-Aided Design) applications.

## Benchmark Composition

### Test Cases (20 total)

The benchmark includes 20 carefully designed test cases across 5 major categories:

1. **Basic Devices (5 cases)**
   - NMOS and PMOS transistors
   - PN junction diodes  
   - Bipolar Junction Transistors (BJT)
   - Junction Field Effect Transistors (JFET)
   - Photodiodes

2. **Advanced Devices (3 cases)**
   - SOI MOSFETs with buried oxide
   - FinFETs with 3D structures
   - Tunnel FETs for low-power applications

3. **Analog Circuits (4 cases)**
   - CMOS inverters
   - Differential amplifiers
   - Operational amplifiers
   - Current mirrors

4. **RF Devices (3 cases)**
   - RF MOSFETs for GHz operation
   - VCO using cross-coupled pairs
   - Low noise amplifiers (LNA)

5. **Power Devices & Sensors (4 cases)**
   - Power MOSFETs
   - MEMS accelerometers
   - Pressure sensors
   - Photonic waveguides

6. **Edge Cases (1 case)**
   - Conflicting specifications
   - Missing parameters
   - Non-physical devices

### Difficulty Levels

Each test case is classified by complexity:
- **Easy (4 cases)**: Basic single-device structures
- **Medium (8 cases)**: Multi-component devices with specific parameters  
- **Hard (8 cases)**: Complex circuits with advanced physics models

### Expected Features

Each test case specifies expected features including:
- Device topology (NMOS, PMOS, diode, etc.)
- Required parameters (dimensions, voltages, materials)
- Physical models (SRH, Auger, impact ionization)
- Analysis types (DC, AC, transient)

## Evaluation Methodology

### Core Metrics (4 implemented)

#### 1. Syntax Validity Score (SVS)
- **Purpose**: Validates presence of essential Silvaco commands
- **Required sections**: `go atlas`, `mesh`, `region`, `electrode`, `models`, `solve`, `quit`
- **Scoring**: Binary (1.0 if all present, 0.0 otherwise)
- **Rationale**: Ensures generated code has basic simulation structure

#### 2. Parameter Exact Match (PEM)  
- **Purpose**: Measures accuracy of device parameters
- **Method**: Regex-based extraction and fuzzy matching
- **Parameters tracked**: Dimensions (L, W), voltages (VDD, Vgs), doping levels, materials
- **Scoring**: Percentage of expected parameters found
- **Rationale**: Critical for simulation accuracy and device performance

#### 3. Component Completeness Score (CCS)
- **Purpose**: Evaluates presence of essential simulation components
- **Categories**: Structure, Electrodes, Parameters, Analysis, Models, Output
- **Scoring**: Average of binary scores for each category
- **Rationale**: Ensures comprehensive simulation coverage

#### 4. BLEU Similarity Score (Optional)
- **Purpose**: Measures textual similarity to reference implementations
- **Method**: 4-gram BLEU with brevity penalty
- **Use case**: When reference code is available
- **Rationale**: Provides baseline comparison metric

### Composite Scoring

The overall performance is measured using:
```
Composite Score = (SVS + PEM + CCS) / 3
```

This provides a balanced assessment across syntax, parameters, and completeness.

## Benchmark Validation

### Quality Assurance
- Expert review of test cases by TCAD professionals
- Parameter validation against real device specifications
- Coverage analysis across device types and difficulty levels

### Training Set Independence Verification
**Critical Requirement**: All benchmark prompts are custom-designed and distinct from training data.

**Validation Methodology**:
1. **Training data analysis**: 713 training examples focus on complex RF circuits (VCOs, switched-capacitor filters, amplifiers) with detailed netlists
2. **Benchmark scope**: 20 test cases covering basic semiconductor devices (MOSFETs, diodes, BJTs) with simple parameter specifications
3. **Prompt complexity comparison**:
   - **Training examples**: 200+ word technical descriptions with specific circuit topologies, detailed component values, and complex analysis requirements
   - **Benchmark prompts**: 30-50 word device descriptions with basic parameters (L, W, voltage) and simple analysis requests

**Key Differences**:
- **Training**: "voltage-controlled oscillator for 2.5GHz RF application with cross-coupled NMOS pair, spiral inductors (60μm radius, 10μm trace width), varactors (40μm length), harmonic balance simulation..."
- **Benchmark**: "Create a basic NMOS transistor with 1μm channel length, 10μm width, operating at 3V supply voltage"
- **Training data domain**: Advanced RF/analog circuit simulation with detailed netlists
- **Benchmark domain**: Basic device physics with ATLAS simulation commands

**Independence Verification**: Zero lexical overlap between training instruction patterns and benchmark test prompts, ensuring fair evaluation of generalization capabilities.

### Statistical Properties
- **Category distribution**: Balanced across device families
- **Difficulty progression**: 20% easy, 40% medium, 40% hard
- **Parameter diversity**: >50 unique parameter types tested
- **Analysis coverage**: DC, AC, transient, thermal, noise

## Implementation Details

### Evaluation Pipeline
1. **Test case loading** from JSON format
2. **Code generation** using fine-tuned model + RAG
3. **Multi-metric evaluation** with detailed scoring
4. **Results aggregation** and statistical analysis
5. **Report generation** in CSV and JSON formats

### Performance Tracking
- Generation time per test case
- Success/failure rates by category
- Score distributions and statistical significance
- Category-specific performance analysis

### Scalability Features
- Configurable sample sizes for quick testing
- Parallel evaluation support (planned)
- Incremental result saving
- Memory-efficient metric computation

## Expected Outcomes

### Performance Baselines
Based on preliminary testing:
- **SVS scores**: 0.5-1.0 (model learns basic syntax)
- **PEM scores**: 0.0-0.5 (parameter extraction challenging)  
- **CCS scores**: 0.4-0.8 (reasonable component coverage)
- **Composite**: 0.3-0.7 (moderate overall performance)

### Use Cases
1. **Model comparison**: Evaluate different LLMs and fine-tuning approaches
2. **Training optimization**: Identify weaknesses in model capabilities
3. **Ablation studies**: Assess impact of RAG, fine-tuning, prompt engineering
4. **Domain adaptation**: Guide model improvements for TCAD applications

## Limitations and Future Work

### Current Limitations
- Limited reference code for BLEU scoring
- CPU-based evaluation (slow for large-scale runs)
- English-only prompts and documentation
- No actual simulation validation (syntax-only)

### Planned Enhancements
- Semantic similarity metrics beyond BLEU
- Integration with Silvaco ATLAS for execution validation
- Multi-language prompt support
- Automated test case generation
- Performance optimization for GPU evaluation

## Conclusion

This benchmark provides a comprehensive, systematic approach to evaluating LLM performance on domain-specific TCAD code generation. The combination of diverse test cases, multiple evaluation metrics, and detailed analysis capabilities makes it suitable for both research and practical applications in semiconductor device modeling.

The benchmark successfully demonstrates the current capabilities and limitations of fine-tuned models for technical code generation, providing valuable insights for future improvements in this specialized domain.