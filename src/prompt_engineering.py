#!/usr/bin/env python3
"""
prompt_engineering.py - Enhanced prompt engineering techniques for SPICE generation
Assignment 4: Implements 4 prompt engineering techniques
"""

import json
from typing import Dict, List, Optional


class PromptEngineeringTechniques:
    """Implementation of 4 prompt engineering techniques for SPICE code generation"""
    
    def __init__(self):
        """Initialize prompt engineering techniques"""
        pass
    
    # =====================================
    # TECHNIQUE 1: CHAIN-OF-THOUGHT (CoT)
    # =====================================
    def chain_of_thought_prompt(self, description: str, retrieved_examples: str = None) -> str:
        """
        Chain-of-Thought: Break down circuit design into logical reasoning steps
        """
        prompt = f"""You are a SPICE circuit expert. Think step by step to design the circuit simulation.

Circuit Request: {description}

Let me think through this step by step:

Step 1: Circuit Analysis
- What type of circuit is this? (amplifier, filter, oscillator, etc.)
- What are the key circuit parameters? (voltages, resistances, capacitances)
- What components are needed? (MOSFETs, resistors, capacitors, etc.)

Step 2: Netlist Strategy
- What nodes need to be defined for proper connectivity?
- How should components be connected? (series, parallel, feedback)
- What is the circuit topology? (common source, differential, etc.)

Step 3: Component Selection
- What device models are needed? (.MODEL statements)
- What component values achieve the specified performance?
- What analysis type matches the request? (.DC, .AC, .TRAN)

Step 4: Code Generation
Now I'll generate the complete SPICE netlist following this reasoning:

Required Structure:
1. Component instances (M1, R1, C1, V1, etc.)
2. Node connections (proper circuit topology)
3. Model definitions (.MODEL statements)
4. Parameter specifications (device dimensions, values)
5. Analysis commands (.DC, .AC, .TRAN, .OP)
6. Output directives (.PROBE, .PRINT)
7. .END statement

Based on my step-by-step analysis, here is the SPICE netlist:

"""
        
        if retrieved_examples:
            prompt += f"\nReference examples for guidance:\n{retrieved_examples}\n"
        
        return prompt

    # =====================================
    # TECHNIQUE 2: FEW-SHOT LEARNING (Enhanced)
    # =====================================
    def few_shot_learning_prompt(self, description: str, retrieved_examples: str = None) -> str:
        """
        Enhanced Few-Shot Learning: Strategic selection of examples with pattern explanation
        """
        # Static high-quality examples covering different circuit types
        static_examples = {
            "nmos_amplifier": {
                "description": "Create a basic NMOS amplifier with 1kΩ load resistor",
                "code": """* Basic NMOS Common-Source Amplifier
M1 vout vin vdd vdd nmos W=10u L=1u
R1 vdd vout 1k
VDD vdd 0 DC 5
VIN vin 0 DC 2.5 AC 1m
CIN vin input 1u
RLOAD vout 0 10k

.model nmos nmos level=1 vto=1 kp=50u
.op
.ac dec 10 1 100meg
.probe ac vm(vout)

.end"""
            },
            "rc_filter": {
                "description": "Create an RC low-pass filter with 1kHz cutoff frequency",
                "code": """* RC Low-Pass Filter
VIN input 0 AC 1 0
R1 input output 1.59k
C1 output 0 100n

.ac dec 10 1 10k
.probe ac vm(output) vp(output)

.end"""
            }
        }
        
        prompt = f"""You are a SPICE circuit expert. Learn from these high-quality examples and generate similar code.

EXAMPLE 1 - NMOS Amplifier Pattern:
Input: {static_examples['nmos_amplifier']['description']}
Output:
{static_examples['nmos_amplifier']['code']}

EXAMPLE 2 - RC Filter Pattern:  
Input: {static_examples['rc_filter']['description']}
Output:
{static_examples['rc_filter']['code']}

PATTERN ANALYSIS:
- All netlists start with component instances and end with ".end"
- Component naming follows SPICE conventions (M1, R1, C1, V1)
- Node connections define circuit topology
- .MODEL statements define device parameters
- Analysis commands match circuit characterization needs (.ac, .dc, .tran)
- .PROBE statements specify output measurements

Now generate SPICE code for this new circuit following the same patterns:

Circuit Request: {description}

Generated SPICE Code:
"""
        
        if retrieved_examples:
            prompt += f"\nAdditional retrieved examples:\n{retrieved_examples}\n"
        
        return prompt

    # =====================================
    # TECHNIQUE 3: PROBLEM DECOMPOSITION
    # =====================================
    def problem_decomposition_prompt(self, description: str, retrieved_examples: str = None) -> str:
        """
        Problem Decomposition: Break complex devices into manageable components
        """
        prompt = f"""You are a Silvaco ATLAS expert. I'll break down the device design into independent components.

Original Request: {description}

DECOMPOSITION APPROACH:
Let me decompose this device into manageable components:

COMPONENT 1: GEOMETRIC STRUCTURE
- Extract device dimensions (length, width, thickness)
- Determine required mesh resolution
- Plan material regions and interfaces

COMPONENT 2: ELECTRICAL CHARACTERISTICS  
- Identify device type and operation mode
- Extract voltage and current specifications
- Determine contact configurations

COMPONENT 3: PHYSICAL PROPERTIES
- Select appropriate material systems
- Determine doping requirements
- Choose relevant physics models

COMPONENT 4: SIMULATION SETUP
- Design appropriate analysis sequence
- Set convergence criteria
- Plan output data extraction

Now I'll implement each component systematically:

IMPLEMENTATION:

Component 1 - Structure:
[Mesh and region definitions based on geometry analysis]

Component 2 - Contacts:
[Electrode placement based on electrical requirements]

Component 3 - Physics:
[Material properties and doping profiles]

Component 4 - Analysis:
[Solution sequence and data extraction]

Complete Silvaco ATLAS code integrating all components:

"""
        
        if retrieved_examples:
            prompt += f"\nReference materials for components:\n{retrieved_examples}\n"
        
        return prompt

    # =====================================
    # TECHNIQUE 4: OUTPUT FORMAT CONTROL
    # =====================================
    def output_format_control_prompt(self, description: str, retrieved_examples: str = None) -> str:
        """
        Output Format Control: Enforce structured Silvaco code generation
        """
        prompt = f"""You are a Silvaco ATLAS expert. Generate code following the EXACT format specification.

Device Request: {description}

MANDATORY OUTPUT FORMAT:
Your response must follow this exact structure with all sections present:

```silvaco
go atlas

# SECTION 1: MESH DEFINITION
[mesh commands - x.mesh and y.mesh with appropriate spacing]

# SECTION 2: MATERIAL REGIONS  
[region commands defining silicon, oxide, and other materials]

# SECTION 3: ELECTRODE PLACEMENT
[electrode commands with proper numbering and naming]

# SECTION 4: DOPING PROFILES
[doping commands creating device physics]

# SECTION 5: PHYSICAL MODELS
[models command with appropriate physics]

# SECTION 6: INITIAL SOLUTION
[solve initial command]

# SECTION 7: ANALYSIS SEQUENCE
[solve commands for device characterization]

# SECTION 8: DATA OUTPUT
[save commands for results]

quit
```

SECTION REQUIREMENTS:
1. Mesh: Must include both x.mesh and y.mesh with spacing adapted to device size
2. Regions: Must define silicon substrate (region 1) and any additional materials
3. Electrodes: Must include all required contacts with proper names
4. Doping: Must create the device junction structure  
5. Models: Must include SRH and Auger for realistic physics
6. Initial: Must include solve initial for starting point
7. Analysis: Must include voltage sweeps or other characterization
8. Output: Must include save commands for data extraction

VALIDATION CHECKLIST:
□ Contains "go atlas" at start
□ Contains "quit" at end  
□ All 8 sections present with comments
□ Proper Silvaco syntax throughout
□ Device-specific parameters included

Generate the complete simulation following this format:

"""
        
        if retrieved_examples:
            prompt += f"\nFormat reference examples:\n{retrieved_examples}\n"
        
        return prompt

    # =====================================
    # BASELINE (Your Current Approach)
    # =====================================
    def baseline_prompt(self, description: str, retrieved_examples: str = None) -> str:
        """
        Baseline: Your current prompt engineering approach for comparison
        """
        prompt = f"""You are a Silvaco ATLAS expert. Generate TCAD simulation code based on the device description.

Device Description: {description}

REQUIRED Silvaco Structure:
1. go atlas
2. Mesh definition (fine mesh for small devices)
3. Material regions (silicon substrate, oxide if MOSFET)
4. Electrode placement (source, drain, gate)
5. Doping profiles (match specified concentrations)
6. Physical models (srh, auger, fermi for heavy doping)
7. Analysis commands (match requested analysis)
8. quit

Generate complete simulation code:
"""
        
        if retrieved_examples:
            prompt += f"\nReference examples:\n{retrieved_examples}\n"
        
        return prompt

# Global instance for easy access
prompt_techniques = PromptEngineeringTechniques()


def get_technique_prompt(technique: str, description: str, retrieved_examples: str = None) -> str:
    """
    Get prompt for specified technique
    
    Args:
        technique: One of 'baseline', 'cot', 'few_shot', 'decomposition', 'format_control'
        description: Device description
        retrieved_examples: RAG retrieved examples (optional)
        
    Returns:
        Formatted prompt string
    """
    techniques = {
        'baseline': prompt_techniques.baseline_prompt,
        'cot': prompt_techniques.chain_of_thought_prompt,
        'few_shot': prompt_techniques.few_shot_learning_prompt,
        'decomposition': prompt_techniques.problem_decomposition_prompt,
        'format_control': prompt_techniques.output_format_control_prompt
    }
    
    if technique not in techniques:
        raise ValueError(f"Unknown technique: {technique}. Available: {list(techniques.keys())}")
    
    return techniques[technique](description, retrieved_examples)


if __name__ == "__main__":
    # Quick test
    test_description = "Create a basic NMOS transistor with 1μm channel length and 10μm width"
    
    print("=== TESTING PROMPT TECHNIQUES ===\n")
    
    for technique in ['baseline', 'cot', 'few_shot', 'decomposition', 'format_control']:
        print(f"--- {technique.upper()} ---")
        prompt = get_technique_prompt(technique, test_description)
        print(f"Prompt length: {len(prompt)} characters")
        print(f"First 200 chars: {prompt[:200]}...")
        print()