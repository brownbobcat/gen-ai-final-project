# Prompt Engineering for SPICE Circuit Code Generation  
**Assignment 4: Advanced Prompt Engineering Techniques**

**Authors:** Nicholas Brown, Nirmit Dagli, Tamuka Manjemu  
**Course:** Generative AI, Fall 2025  
**Date:** December 12, 2025  

---

## Abstract

This report investigates four advanced prompt engineering techniques applied to SPICE circuit code generation using our fine-tuned Qwen2-0.5B model. We compare Chain-of-Thought (CoT), enhanced Few-Shot Learning, Problem Decomposition, and Output Format Control against our baseline approach across 8 representative test cases. Results show that **Few-Shot Learning** achieves the highest performance with **0.594** composite score, representing a **152%** improvement over the baseline approach (0.236). The findings demonstrate the effectiveness of structured prompting for domain-specific circuit netlist generation tasks.

---

## 1. Introduction

### 1.1 Background
Large language models have shown remarkable capabilities in code generation, but their performance heavily depends on how tasks are presented through prompts. In our final project, we developed a fine-tuned system for generating SPICE circuit netlists from natural language descriptions. This assignment extends that work by exploring advanced prompt engineering techniques to improve generation quality and consistency.

### 1.2 Motivation
SPICE circuit code generation presents unique challenges requiring:
- **Precise parameter extraction** from natural language to numerical values
- **Structured output format** following SPICE netlist syntax
- **Domain knowledge integration** of electronic circuit design principles
- **Adaptive complexity** handling from basic components to complex circuits

Standard prompting approaches may not fully leverage the model's capabilities for such specialized tasks, motivating the exploration of advanced techniques.

### 1.3 Research Questions
1. Which prompt engineering techniques are most effective for SPICE circuit code generation?
2. How do different techniques impact specific evaluation metrics (syntax, parameters, completeness)?
3. What are the trade-offs between prompt complexity and generation quality?
4. How can these techniques be integrated into practical circuit design automation workflows?

---

## 2. Methodology

### 2.1 Technique Selection and Rationale

We selected four advanced prompt engineering techniques based on their relevance to our TCAD domain:

#### 2.1.1 Chain-of-Thought (CoT)
**Rationale:** SPICE circuit simulation requires systematic reasoning through circuit topology, component values, and analysis requirements. CoT prompting encourages the model to break down complex circuit descriptions into logical steps.

**Implementation:** The prompt guides the model through four reasoning stages:
1. Circuit Analysis (topology, components, parameters)
2. Netlist Strategy (nodes, connections, hierarchy)
3. Component Selection (models, values, analysis)
4. Code Generation (structured implementation)

#### 2.1.2 Enhanced Few-Shot Learning
**Rationale:** Our RAG system already provides examples, but strategic selection and pattern explanation can improve learning. We enhance few-shot learning with carefully curated examples and explicit pattern analysis.

**Implementation:** Provides high-quality static examples for different circuit types (NMOS amplifier, RC filter) with detailed pattern explanations covering structure, syntax, and adaptation principles.

#### 2.1.3 Problem Decomposition
**Rationale:** Complex SPICE circuits involve multiple interacting components (topology, devices, analysis). Decomposing problems into manageable components allows systematic handling of complexity.

**Implementation:** Breaks circuit design into four independent components:
1. Circuit Structure (topology, nodes, connections)
2. Component Specifications (devices, values, models)
3. Electrical Properties (voltages, currents, parameters)
4. Analysis Setup (simulation type, convergence, outputs)

#### 2.1.4 Output Format Control
**Rationale:** SPICE netlists have strict syntax requirements with mandatory elements. Enforcing structured output format ensures syntactic validity and completeness.

**Implementation:** Provides explicit formatting requirements with essential SPICE sections, validation checklist, and structured code blocks with comments.

### 2.2 Experimental Design

#### 2.2.1 Test Case Selection
We selected 8 representative test cases from our benchmark covering diverse circuit categories:
- **Basic circuits:** NMOS amplifier (spice_nmos_01), Diode circuit (spice_diode_01)
- **Advanced circuits:** Power MOSFET (mosfet_04), Short-channel NMOS (mosfet_03)
- **Specialized circuits:** BJT amplifier (bjt_01), Photonic device (photonic_01)
- **Complex systems:** MEMS accelerometer (sensor_02)
- **Edge cases:** Conflicting specifications (edge_01)

#### 2.2.2 Evaluation Metrics
We use the same comprehensive evaluation framework from our final project:
- **SVS (Syntax Validity Score):** Binary validation of required SPICE sections
- **PEM (Parameter Exact Match):** Accuracy of numerical parameter extraction
- **CCS (Component Completeness Score):** Coverage of essential circuit components
- **Composite Score:** Average of SVS, PEM, and CCS

#### 2.2.3 Experimental Conditions
- **Model:** Fine-tuned Qwen2-0.5B with LoRA adapters
- **Generation settings:** Temperature=0.1, max_length=2048
- **RAG:** Disabled for fair technique comparison
- **Repetitions:** Single run per technique/case combination
- **Total experiments:** 5 techniques × 8 test cases = 40 experiments

---

## 3. Implementation

### 3.1 Prompt Template Design

Each prompt template was designed to leverage specific cognitive strategies for improved TCAD code generation. The templates range from 709 characters (baseline) to 2,660 characters (few-shot), with varying levels of structure and guidance.

#### 3.1.1 Baseline Template
```
You are a SPICE circuit expert. Generate SPICE netlist code based on the circuit description.

Circuit Description: {description}

REQUIRED SPICE Structure:
1. Component instances (M1, R1, C1, L1, V1, I1)
2. Node connections (proper netlist topology)
3. Model definitions (.MODEL statements)
4. Parameter specifications (device dimensions, values)
5. Analysis commands (.DC, .AC, .TRAN, .OP)
6. Output directives (.PROBE, .PRINT)
7. .END statement

Generate complete SPICE netlist:
```

#### 3.1.2 Chain-of-Thought Template
Guides the model through systematic 4-step reasoning:
```
Think step by step to design the device simulation:
Step 1: Device Analysis (type, parameters, materials)
Step 2: Simulation Strategy (mesh, regions, contacts) 
Step 3: Physics Selection (models, doping, analysis)
Step 4: Code Generation (structured implementation)

Based on my step-by-step analysis, here is the simulation code:
```
**Length**: 1,516 characters | **Strategy**: Sequential reasoning

#### 3.1.3 Enhanced Few-Shot Learning Template  
Provides curated examples with explicit pattern analysis:
```
Learn from these high-quality examples:

EXAMPLE 1 - Basic NMOS Pattern:
Input: Create a basic NMOS with 1μm channel length...
Output: [Complete Silvaco code with go atlas, mesh, regions...]

PATTERN ANALYSIS:
- All simulations start with "go atlas" and end with "quit"
- Mesh density adapts to device dimensions
- Material regions define device structure
- Analysis commands match device characterization needs

Now generate code following the same patterns:
```
**Length**: 2,660 characters | **Strategy**: Learning by example with pattern extraction

#### 3.1.4 Problem Decomposition Template
Systematically breaks complex devices into manageable components:
```
I'll decompose this device into independent components:

COMPONENT 1: GEOMETRIC STRUCTURE
COMPONENT 2: ELECTRICAL CHARACTERISTICS  
COMPONENT 3: PHYSICAL PROPERTIES
COMPONENT 4: SIMULATION SETUP

Implementation:
Component 1 - Structure: [Mesh and region definitions]
Component 2 - Contacts: [Electrode placement]
Component 3 - Physics: [Material properties and doping]
Component 4 - Analysis: [Solution sequence]
```
**Length**: 1,411 characters | **Strategy**: Systematic component-wise design

#### 3.1.5 Output Format Control Template
Enforces strict structural requirements with validation:
```
Generate code following EXACT format specification:

MANDATORY OUTPUT FORMAT:
```silvaco
go atlas
# SECTION 1: MESH DEFINITION
# SECTION 2: MATERIAL REGIONS
# SECTION 3: ELECTRODE PLACEMENT
# SECTION 4: DOPING PROFILES
# SECTION 5: PHYSICAL MODELS
# SECTION 6: INITIAL SOLUTION
# SECTION 7: ANALYSIS SEQUENCE
# SECTION 8: DATA OUTPUT
quit
```

VALIDATION CHECKLIST:
□ Contains "go atlas" at start
□ All 8 sections present with comments
□ Proper Silvaco syntax throughout
```
**Length**: 1,845 characters | **Strategy**: Structural enforcement with validation

### 3.2 Experimental Framework
We implemented a comprehensive experimental framework (`prompt_experiments.py`) that:
- Loads test cases from our existing benchmark
- Applies each prompt technique systematically
- Measures generation time and success rates
- Evaluates outputs using our established metrics
- Stores detailed results for analysis

---

## 4. Results

### 4.1 Overall Performance Comparison

| Technique | Composite Score | SVS Score | PEM Score | CCS Score | Generation Time | Success Rate |
|-----------|----------------|-----------|-----------|-----------|-----------------|--------------|
| Baseline | 0.236 | 0.000 | 0.042 | 0.667 | 46.9 s | 100% |
| Chain-of-Thought | 0.076 | 0.000 | 0.000 | 0.229 | 44.5 s | 100% |
| **Few-Shot Learning** | **0.594** | **0.750** | **0.135** | **0.896** | **54.9 s** | **100%** |
| Problem Decomposition | 0.194 | 0.000 | 0.083 | 0.500 | 43.8 s | 100% |
| Output Format Control | 0.229 | 0.000 | 0.062 | 0.625 | 50.2 s | 100% |

**Key Findings:**
- **Best performing technique**: Few-Shot Learning with 0.594 composite score
- **Performance improvement**: 152% improvement over baseline (0.594 vs 0.236)
- **Strongest metric**: Syntax Validity (SVS) showed most significant improvement (75% success rate)
- **Component completeness**: Few-Shot achieved excellent coverage (0.896 CCS score)
- **Trade-off**: Few-Shot requires 17% longer generation time but delivers 2.5x better results

### 4.2 Metric-Specific Analysis

#### 4.2.1 Syntax Validity Score (SVS)
Few-Shot Learning dramatically outperformed all other techniques in syntax validity:
- **Few-Shot**: 0.750 (75% of generated codes had perfect Silvaco syntax)
- **All others**: 0.000 (no other technique achieved valid syntax structure)

This demonstrates that providing concrete examples with pattern explanations is crucial for learning domain-specific syntax requirements.

#### 4.2.2 Parameter Exact Match (PEM)  
Parameter extraction accuracy was challenging for all techniques:
- **Few-Shot**: 0.135 (best performance, extracted 13.5% of expected parameters)
- **Problem Decomposition**: 0.083 (systematic approach helped somewhat)
- **Baseline**: 0.042 (minimal parameter extraction)
- **Other techniques**: 0.000-0.062 (poor parameter mapping)

This reveals that numerical parameter extraction from natural language remains a significant challenge requiring specialized approaches.

#### 4.2.3 Component Completeness Score (CCS)
Few-Shot Learning excelled at including essential simulation components:
- **Few-Shot**: 0.896 (nearly complete simulation coverage)
- **Baseline**: 0.667 (reasonable but incomplete)
- **Format Control**: 0.625 (structured approach helped)
- **Other techniques**: 0.229-0.500 (incomplete simulations)

### 4.3 Category-Specific Performance

Analysis across 8 test cases spanning 5 device categories reveals technique-specific strengths:

| Category | Baseline | CoT | Few-Shot | Decomposition | Format Control |
|----------|----------|-----|----------|---------------|-----------------|
| **Basic MOSFET** | 0.278 | 0.056 | 0.611 | 0.139 | 0.250 |
| **Advanced MOSFET** | 0.222 | 0.056 | 0.556 | 0.139 | 0.222 |
| **Diode** | 0.222 | 0.056 | 0.556 | 0.222 | 0.222 |
| **BJT** | 0.222 | 0.056 | 0.556 | 0.222 | 0.222 |
| **Photonic** | 0.222 | 0.056 | 0.556 | 0.222 | 0.222 |
| **Sensor (MEMS)** | 0.222 | 0.056 | 0.556 | 0.222 | 0.222 |
| **Power Device** | 0.278 | 0.156 | 0.667 | 0.278 | 0.306 |
| **Edge Case** | 0.222 | 0.056 | 0.556 | 0.139 | 0.222 |

**Key Observations:**
- **Few-Shot Learning** consistently outperformed across all categories
- **Power devices** showed higher scores across techniques (more structured specifications)
- **Edge cases** exposed technique limitations equally
- **Complex devices** (photonic, MEMS) challenged all approaches similarly

### 4.4 Example Outputs

#### 4.4.1 Test Case: Basic NMOS (mosfet_01)
**Description:** "Create a basic NMOS transistor with 1μm channel length, 10μm width, operating at 3V supply voltage with silicon substrate and silicon dioxide gate insulator. Include basic DC characteristics analysis."

**Baseline Output (Composite Score: 0.278):**
```silvaco
# Device description comments (repetitive, no actual Silvaco code)
# Define the material types used in this example
# - Silicon (Si) for the active region
# - Oxide (Ox) for the gate oxide
# [Continues with extensive comments but no executable code]
```

**Few-Shot Learning Output (Composite Score: 0.611):**
```silvaco
go atlas

# Mesh for 1μm device
mesh space.mult=1.0
x.mesh l=-0.5 spac=0.01
x.mesh l=0.0 spac=0.002
x.mesh l=1.0 spac=0.002
y.mesh l=0.0 spac=0.001
y.mesh l=0.01 spac=0.002

# Silicon substrate and gate oxide
region num=1 material=silicon x.min=-0.5 x.max=1.5 y.min=0.0 y.max=0.5
region num=2 material=oxide x.min=0.0 x.max=1.0 y.min=0.0 y.max=0.01

# Electrodes
electrode num=1 name=source x.min=-0.5 x.max=0.0 y.min=0.0 y.max=0.0
electrode num=2 name=drain x.min=1.0 x.max=1.5 y.min=0.0 y.max=0.0
electrode num=3 name=gate x.min=0.0 x.max=1.0 y.min=0.01 y.max=0.01

# Doping profiles
doping uniform p.type conc=1e15 region=1
doping gaussian n.type conc=1e17 char.length=0.05

# Models and analysis
models srh auger fermi
solve initial
solve vgate=3.0 vstep=0.1 name=gate

quit
```

**Analysis:** Few-Shot Learning produced executable Silvaco code with proper structure, mesh definition, material regions, electrodes, and doping profiles. The baseline generated only comments without actual simulation commands, demonstrating the critical importance of concrete examples for domain-specific code generation.

---

## 5. Analysis and Discussion

### 5.1 Technique Effectiveness

#### 5.1.1 Most Effective Technique
**Few-Shot Learning** emerged as the clear winner with a 0.594 composite score, achieving:
- **Superior syntax validity**: Only technique to generate valid Silvaco structure (75% success)
- **Excellent completeness**: 89.6% component coverage vs 66.7% baseline
- **Consistent performance**: Outperformed across all 8 test cases and device categories
- **Learning by example**: Concrete patterns proved more effective than abstract instructions

**Success factors:**
1. **Concrete examples** provided syntactic templates the model could follow
2. **Pattern analysis** explicitly taught structural requirements
3. **Diverse examples** (NMOS + diode) covered key device types
4. **Length optimization** (2,660 chars) balanced detail with computational efficiency

#### 5.1.2 Technique Strengths and Weaknesses

**Few-Shot Learning:**
- ✅ Strengths: Best syntax validity, highest completeness, consistent across categories
- ⚠️ Weaknesses: Longest generation time (54.9s), still poor parameter extraction

**Baseline:**
- ✅ Strengths: Moderate component coverage (66.7%), reasonable generation time
- ⚠️ Weaknesses: No valid syntax generation, minimal parameter extraction

**Output Format Control:**
- ✅ Strengths: Structured approach, reasonable completeness (62.5%)
- ⚠️ Weaknesses: Failed to achieve valid syntax despite explicit formatting requirements

**Problem Decomposition:**
- ✅ Strengths: Systematic approach, moderate performance on complex devices
- ⚠️ Weaknesses: Component isolation may have disrupted code coherence

**Chain-of-Thought:**
- ✅ Strengths: Shortest generation time (44.5s), logical reasoning process
- ⚠️ Weaknesses: Worst overall performance, abstract reasoning didn't translate to syntax

### 5.2 Impact on Evaluation Metrics

#### 5.2.1 Syntax Structure Improvements
The dramatic difference in Syntax Validity Scores reveals the critical importance of concrete examples:
- **Few-Shot Learning**: 0.750 (75% valid syntax) through explicit pattern teaching
- **All other techniques**: 0.000 (no valid syntax generated)

This suggests that abstract instructions, even with detailed requirements, cannot substitute for concrete syntactic templates in domain-specific code generation.

#### 5.2.2 Parameter Extraction Enhancements
Parameter extraction remained challenging across all techniques:
- **Best performance**: Few-Shot (13.5%) through example-based parameter patterns
- **Moderate success**: Problem Decomposition (8.3%) via systematic parameter isolation
- **Poor results**: Other techniques (0-6.2%) failed to learn parameter mapping

The low scores indicate that natural language to numerical parameter mapping requires specialized approaches beyond general prompt engineering.

#### 5.2.3 Completeness Gains
Component completeness showed significant improvement with structured approaches:
- **Few-Shot Learning**: 89.6% through comprehensive example templates
- **Output Format Control**: 62.5% via explicit section requirements
- **Baseline**: 66.7% showing reasonable natural capability
- **Other techniques**: 22.9-50.0% suggesting disruption from complex prompts

Few-Shot's success demonstrates that showing complete examples is more effective than describing completeness requirements.

### 5.3 Practical Implications

#### 5.3.1 Integration with Final Project
The Few-Shot Learning technique can immediately enhance our final project system:

**Immediate Implementation:**
- Replace baseline prompts with Few-Shot templates for 152% performance improvement
- Integrate curated examples into the RAG system for hybrid example retrieval
- Add pattern analysis explanations to help the model learn structural requirements

**System Enhancement:**
- **RAG + Few-Shot synergy**: Combine retrieved examples with static high-quality templates
- **Adaptive examples**: Select Few-Shot templates based on device category
- **Quality gates**: Use syntax validity checks to trigger Few-Shot prompts for failed generations

#### 5.3.2 Scalability Considerations
**Performance vs. Cost Trade-offs:**
- **Few-Shot**: 17% longer generation time but 2.5x better results → Excellent ROI
- **Prompt length**: 2,660 characters manageable for production systems
- **Token costs**: Additional input tokens justified by reduced regeneration needs

**Production Deployment:**
- Batch processing can amortize longer generation times
- High success rate (75% valid syntax) reduces manual post-processing
- Template maintenance overhead minimal for stable domain like Silvaco ATLAS

#### 5.3.3 Domain Adaptation Insights
**Key lessons for technical domain prompt engineering:**

1. **Concrete > Abstract**: Show examples rather than describe requirements
2. **Domain patterns matter**: Technical syntax requires explicit pattern teaching
3. **Length optimization**: Detailed examples (2,660 chars) outperform concise instructions
4. **Example diversity**: Multiple device types in templates improve generalization
5. **Pattern explanation**: Explicitly teaching "why" patterns work enhances learning

**Generalization to other technical domains:**
- Medical diagnosis code generation
- Legal document automation  
- Scientific protocol generation
- Financial modeling scripts

---

## 6. Conclusions and Future Work

### 6.1 Key Findings
1. **Few-Shot Learning** achieved 152% improvement in composite score (0.594 vs 0.236 baseline)
2. **Concrete examples are crucial**: Only Few-Shot achieved valid Silvaco syntax (75% success rate)
3. **Parameter extraction remains challenging**: Best performance was only 13.5% parameter accuracy
4. **Structured prompting works**: Few-Shot achieved 89.6% component completeness vs 66.7% baseline

### 6.2 Contributions to Final Project
This prompt engineering study enhances our final project by:
- **Improving generation quality** through Few-Shot Learning (152% improvement)
- **Providing systematic evaluation** of 5 prompting approaches across 8 test cases
- **Demonstrating domain-specific optimization** for TCAD applications
- **Establishing best practices**: Concrete examples > abstract instructions for technical domains

### 6.3 Limitations
- Limited to 8 test cases for computational efficiency
- Single model evaluation (Qwen2-0.5B)
- No human expert evaluation of generated code quality
- Static prompt templates without adaptive optimization

### 6.4 Future Directions
1. **Dynamic prompt adaptation** based on device complexity
2. **Multi-turn conversation** for iterative code refinement
3. **Hybrid techniques** combining multiple approaches
4. **Expert evaluation** of code quality beyond automated metrics
5. **Cross-model validation** with different base models

---

## 7. References

1. Wei, J., et al. (2022). Chain-of-thought prompting elicits reasoning in large language models. *Advances in Neural Information Processing Systems*.

2. Brown, T., et al. (2020). Language models are few-shot learners. *Advances in Neural Information Processing Systems*.

3. Zhou, D., et al. (2022). Least-to-most prompting enables complex reasoning in large language models. *arXiv preprint arXiv:2205.10625*.

4. [Course materials and lectures on prompt engineering techniques]

5. [Silvaco ATLAS documentation and simulation examples]

