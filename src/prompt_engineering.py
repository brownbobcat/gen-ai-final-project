#!/usr/bin/env python3
"""
prompt_engineering.py - Enhanced prompt engineering techniques for TCAD generation
Assignment 4: Implements 4 prompt engineering techniques
"""

import json
from typing import Dict, List, Optional


class PromptEngineeringTechniques:
    """Implementation of 4 prompt engineering techniques for TCAD code generation"""
    
    def __init__(self):
        """Initialize prompt engineering techniques"""
        pass
    
    # =====================================
    # TECHNIQUE 1: CHAIN-OF-THOUGHT (CoT)
    # =====================================
    def chain_of_thought_prompt(self, description: str, retrieved_examples: str = None) -> str:
        """
        Chain-of-Thought: Break down device design into logical reasoning steps
        """
        prompt = f"""You are a Silvaco ATLAS expert. Think step by step to design the device simulation.

Device Request: {description}

Let me think through this step by step:

Step 1: Device Analysis
- What type of device is this? (MOSFET, diode, BJT, etc.)
- What are the key physical parameters? (dimensions, doping, voltages)
- What material system should I use? (Si, SiO2, etc.)

Step 2: Simulation Strategy
- What mesh density is needed based on device dimensions?
- Which regions need to be defined? (substrate, gate oxide, etc.)
- Where should electrodes be placed for proper contact?

Step 3: Physics Selection
- What physical models are needed? (SRH, Auger, quantum effects)
- What doping profiles will achieve the specified characteristics?
- What analysis type matches the request? (DC, AC, transient)

Step 4: Code Generation
Now I'll generate the complete Silvaco ATLAS code following this reasoning:

Required Structure:
1. go atlas (start simulation)
2. Mesh definition (adaptive to device size)
3. Material regions (match device structure)
4. Electrode placement (proper contact geometry)
5. Doping profiles (achieve target concentrations)
6. Physical models (match device physics)
7. Analysis commands (solve for requested characteristics)
8. quit (end simulation)

Based on my step-by-step analysis, here is the simulation code:

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
        # Static high-quality examples covering different device types
        static_examples = {
            "nmos_basic": {
                "description": "Create a basic NMOS with 1μm channel length and 10μm width",
                "code": """go atlas

# Mesh for 1μm device
mesh space.mult=1.0
x.mesh l=-0.5 spac=0.01
x.mesh l=0.0 spac=0.002
x.mesh l=1.0 spac=0.002
x.mesh l=1.5 spac=0.01
y.mesh l=0.0 spac=0.001
y.mesh l=0.01 spac=0.002
y.mesh l=0.5 spac=0.05

# Silicon substrate and gate oxide
region num=1 material=silicon x.min=-0.5 x.max=1.5 y.min=0.0 y.max=0.5
region num=2 material=oxide x.min=0.0 x.max=1.0 y.min=0.0 y.max=0.01

# Electrodes
electrode num=1 name=source x.min=-0.5 x.max=0.0 y.min=0.0 y.max=0.0
electrode num=2 name=drain x.min=1.0 x.max=1.5 y.min=0.0 y.max=0.0
electrode num=3 name=gate x.min=0.0 x.max=1.0 y.min=0.01 y.max=0.01

# Doping profiles
doping uniform p.type conc=1e15 region=1
doping gaussian n.type conc=1e17 char.length=0.05 x.min=-0.5 x.max=0.0
doping gaussian n.type conc=1e17 char.length=0.05 x.min=1.0 x.max=1.5

# Models and analysis
models srh auger fermi
solve initial
solve vgate=3.0 vstep=0.1 name=gate
save outfile=nmos_1u.str

quit"""
            },
            
            "diode_basic": {
                "description": "Design a p-n junction diode with 10μm junction depth",
                "code": """go atlas

# Mesh for diode structure
mesh space.mult=1.0
x.mesh l=0.0 spac=0.005
x.mesh l=10.0 spac=0.01
x.mesh l=20.0 spac=0.01
y.mesh l=0.0 spac=0.002
y.mesh l=0.5 spac=0.05

# Silicon region
region num=1 material=silicon x.min=0.0 x.max=20.0 y.min=0.0 y.max=0.5

# Electrodes
electrode num=1 name=anode x.min=0.0 x.max=10.0 y.min=0.0 y.max=0.0
electrode num=2 name=cathode x.min=10.0 x.max=20.0 y.min=0.0 y.max=0.0

# P-N junction doping
doping uniform n.type conc=1e15 region=1
doping uniform p.type conc=1e16 x.min=0.0 x.max=10.0

# Models and I-V analysis
models srh auger
solve initial
solve vanode=1.0 vstep=0.1 name=anode
save outfile=diode.str

quit"""
            }
        }
        
        prompt = f"""You are a Silvaco ATLAS expert. Learn from these high-quality examples and generate similar code.

EXAMPLE 1 - Basic NMOS Pattern:
Input: {static_examples['nmos_basic']['description']}
Output:
{static_examples['nmos_basic']['code']}

EXAMPLE 2 - Basic Diode Pattern:  
Input: {static_examples['diode_basic']['description']}
Output:
{static_examples['diode_basic']['code']}

PATTERN ANALYSIS:
- All simulations start with "go atlas" and end with "quit"
- Mesh density adapts to device dimensions (finer for smaller features)
- Material regions define device structure (silicon + oxides)
- Electrodes match device type (3 for MOSFETs, 2 for diodes)
- Doping creates the device physics (p/n junctions)
- Models include SRH and Auger for realistic physics
- Analysis commands match the device characterization needs

Now generate code for this new device following the same patterns:

Device Request: {description}

Generated Silvaco Code:
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