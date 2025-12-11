#!/usr/bin/env python3
"""
enhanced_prompts.py - Enhanced SPICE prompting system with Few-Shot Learning
Uses professional SPICE netlist templates with dynamic parameter extraction and curated examples
"""

import re
from typing import Dict, List, Optional, Tuple


class EnhancedSPICEPrompts:
    """Enhanced prompting system using Few-Shot Learning for SPICE netlists"""
    
    def __init__(self):
        """Initialize enhanced prompt system with dynamic SPICE examples"""
        # Dynamic SPICE examples that adapt based on extracted parameters
        self.example_templates = {
            'nmos': {
                'description': 'NMOS transistor characterization',
                'template': '''* {description}
M1 vd vg 0 0 {model_name} L={length} W={width}
VDD vd 0 DC {vdd}
VGS vg 0 DC 0
VSS 0 0 DC 0

.MODEL {model_name} NMOS(
+ LEVEL=1 VTO=0.7 KP=120u GAMMA=0.5 PHI=0.6
+ LAMBDA=0.1 RD=10 RS=10 CBD=5f CBS=5f
+ CGDO=0.4n CGSO=0.4n CGBO=0.4n)

{analysis_command}
.PROBE DC I(VDD)
.END'''
            },
            
            'pmos': {
                'description': 'PMOS transistor characterization',
                'template': '''* {description}
M1 0 vg vd vd {model_name} L={length} W={width}
VDD vd 0 DC {vdd}
VGS vg 0 DC 0
VSS 0 0 DC 0

.MODEL {model_name} PMOS(
+ LEVEL=1 VTO=-0.7 KP=50u GAMMA=0.5 PHI=0.6
+ LAMBDA=0.1 RD=15 RS=15 CBD=5f CBS=5f
+ CGDO=0.4n CGSO=0.4n CGBO=0.4n)

{analysis_command}
.PROBE DC I(VDD)
.END'''
            },
            
            'bjt': {
                'description': 'BJT characterization',
                'template': '''* {description}
Q1 vc vb 0 {model_name}
VCE vc 0 DC {vdd}
VBE vb 0 DC 0
VEE 0 0 DC 0

.MODEL {model_name} NPN(
+ IS=1e-16 BF=100 BR=1 VAF=100 VAR=10
+ RB=10 RC=1 RE=0.5 CJE=0.5p CJC=0.3p)

{analysis_command}
.PROBE DC I(VCE) I(VBE)
.END'''
            },
            
            'capacitor': {
                'description': 'Capacitor characterization',
                'template': '''* {description}
C1 vg vsub {capacitance}
VG vg 0 DC 0
VSUB vsub 0 DC 0

{analysis_command}
.PROBE AC C(C1)
.END'''
            },
            
            'resistor_divider': {
                'description': 'Resistor divider circuit',
                'template': '''* {description}
VIN vin 0 DC {vdd}
R1 vin vout {r1_value}
R2 vout 0 {r2_value}

{analysis_command}
.PROBE DC V(vout) I(VIN)
.END'''
            },
            
            'resistor': {
                'description': 'Resistor circuit',
                'template': '''* {description}
VIN vin 0 DC {vdd}
R1 vin vout {r1_value}
RLOAD vout 0 {r2_value}

{analysis_command}
.PROBE DC V(vout) I(VIN)
.END'''
            },
            
            'rc_circuit': {
                'description': 'RC circuit',
                'template': '''* {description}
VIN vin 0 DC {vdd}
R1 vin vout {r1_value}
C1 vout 0 {capacitance}

{analysis_command}
.PROBE DC V(vout) I(VIN)
.END'''
            },
            
            'lc_circuit': {
                'description': 'LC circuit',
                'template': '''* {description}
VIN vin 0 DC {vdd}
L1 vin vout {inductance}
C1 vout 0 {capacitance}

{analysis_command}
.PROBE DC V(vout) I(VIN)
.END'''
            },
            
            'rlc_circuit': {
                'description': 'RLC circuit',
                'template': '''* {description}
VIN vin 0 DC {vdd}
R1 vin vn1 {r1_value}
L1 vn1 vout {inductance}
C1 vout 0 {capacitance}

{analysis_command}
.PROBE DC V(vout) I(VIN)
.END'''
            },
            
            'generic': {
                'description': 'Generic circuit',
                'template': '''* {description}
* Add your circuit components here
VIN vin 0 DC {vdd}

{analysis_command}
.PROBE DC V(vin)
.END'''
            }
        }
    
    def extract_parameters(self, description: str) -> Dict[str, str]:
        """Extract SPICE-relevant parameters dynamically from description"""
        params = {}
        desc_lower = description.lower()
        
        # Channel length extraction for SPICE L parameter
        l_patterns = [
            r'L\s*=\s*(\d+\.?\d*)\s*([μµun])',
            r'(\d+\.?\d*)\s*([μµun]m?)\s+(?:channel|gate)\s+length',
            r'(?:channel|gate)\s+length.*?(\d+\.?\d*)\s*([μµun]m?)',
            r'(\d+)\s*nm.*?(?:channel|gate)',
        ]
        
        for pattern in l_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                if len(match.groups()) > 1 and match.group(2):
                    unit = match.group(2).lower()
                    if 'n' in unit:
                        params['length'] = f"{int(value) if value == int(value) else value}n"
                    else:
                        params['length'] = f"{int(value) if value == int(value) else value}u"
                break
        
        # Width extraction for SPICE W parameter
        w_patterns = [
            r'W\s*=\s*(\d+\.?\d*)\s*([μµun])',
            r'(\d+\.?\d*)\s*([μµun]m?)\s+width',
            r'width.*?(\d+\.?\d*)\s*([μµun]m?)',
        ]
        
        for pattern in w_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                if len(match.groups()) > 1 and match.group(2):
                    unit = match.group(2).lower()
                    if 'n' in unit:
                        params['width'] = f"{int(value) if value == int(value) else value}n"
                    else:
                        params['width'] = f"{int(value) if value == int(value) else value}u"
                break
        
        # Voltage extraction
        v_patterns = [
            r'(\d+\.?\d*)\s*V\s+(?:supply|voltage|VDD|operation)',
            r'VDD\s*=\s*(\d+\.?\d*)\s*V?',
            r'supply.*?(\d+\.?\d*)\s*V',
            r'(\d+\.?\d*)\s*V.*?(?:supply|voltage|power)',
        ]
        
        for pattern in v_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                params['vdd'] = f"{value}" if value != int(value) else f"{int(value)}"
                break
        
        # Capacitance extraction
        c_patterns = [
            r'C\d*\s*=\s*(\d+\.?\d*)\s*([pnumk]?F)',
            r'(\d+\.?\d*)\s*([pnumk]?F)',
            r'capacitance.*?(\d+\.?\d*)\s*([pnumk]?F)',
            r'capacitor.*?(\d+\.?\d*)\s*([pnumk]?F)',
        ]
        
        for pattern in c_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                unit = match.group(2) if len(match.groups()) > 1 else 'F'
                params['capacitance'] = f"{int(value) if value == int(value) else value}{unit.lower()}"
                break
        
        # Inductance extraction (avoid confusion with transistor channel length)
        # Only extract inductance if we're not talking about transistors
        if not any(term in description.lower() for term in ['transistor', 'mosfet', 'channel', 'gate', 'drain', 'source']):
            l_patterns = [
                r'L\d*\s*=\s*(\d+\.?\d*)\s*([pnumkm]?H)',
                r'(\d+\.?\d*)\s*([pnumkm]?H)',
                r'inductance.*?(\d+\.?\d*)\s*([pnumkm]?H)',
                r'inductor.*?(\d+\.?\d*)\s*([pnumkm]?H)',
            ]
            
            for pattern in l_patterns:
                match = re.search(pattern, description, re.IGNORECASE)
                if match:
                    value = float(match.group(1))
                    unit = match.group(2) if len(match.groups()) > 1 else 'H'
                    params['inductance'] = f"{int(value) if value == int(value) else value}{unit.lower()}"
                    break
        
        # Resistor value extraction
        r_patterns = [
            r'R1\s*=\s*(\d+\.?\d*)\s*([kmMGT]?Ω|[kmMGT]?ohm|[kmMGT])',
            r'R2\s*=\s*(\d+\.?\d*)\s*([kmMGT]?Ω|[kmMGT]?ohm|[kmMGT])',
            r'(\d+\.?\d*)\s*([kmMGT]?Ω|[kmMGT]?ohm|[kmMGT])',
        ]
        
        r1_found = False
        r2_found = False
        
        # Look for R1 specifically - handle "R1 = 10kΩ" pattern
        r1_match = re.search(r'R1\s*=\s*(\d+\.?\d*)\s*k?Ω?', description, re.IGNORECASE)
        if r1_match:
            value = float(r1_match.group(1))
            # Check if it's followed by 'k' in the description around this position
            full_match = r1_match.group(0)
            if 'k' in full_match.lower():
                params['r1_value'] = f"{int(value) if value == int(value) else value}k"
            else:
                params['r1_value'] = f"{int(value) if value == int(value) else value}"
            r1_found = True
        
        # Look for R2 specifically - handle "R2 = 10kΩ" pattern  
        r2_match = re.search(r'R2\s*=\s*(\d+\.?\d*)\s*k?Ω?', description, re.IGNORECASE)
        if r2_match:
            value = float(r2_match.group(1))
            # Check if it's followed by 'k' in the description around this position
            full_match = r2_match.group(0)
            if 'k' in full_match.lower():
                params['r2_value'] = f"{int(value) if value == int(value) else value}k"
            else:
                params['r2_value'] = f"{int(value) if value == int(value) else value}"
            r2_found = True
        
        # If we didn't find specific R1/R2, look for general resistor values
        if not r1_found or not r2_found:
            resistor_values = []
            for match in re.finditer(r'(\d+\.?\d*)\s*([kmMGT]?Ω|[kmMGT]?ohm|[kmMGT])', description, re.IGNORECASE):
                value = float(match.group(1))
                unit = match.group(2).lower() if len(match.groups()) > 1 else ''
                if 'k' in unit:
                    resistor_values.append(f"{value}k")
                elif 'm' in unit.lower() and 'meg' not in unit.lower():
                    resistor_values.append(f"{value}meg")
                else:
                    resistor_values.append(f"{int(value) if value == int(value) else value}")
            
            if resistor_values:
                if not r1_found and len(resistor_values) > 0:
                    params['r1_value'] = resistor_values[0]
                if not r2_found and len(resistor_values) > 1:
                    params['r2_value'] = resistor_values[1]
                elif not r2_found and len(resistor_values) == 1:
                    params['r2_value'] = resistor_values[0]  # Same value for both
        
        return params
    
    def detect_device_type(self, description: str) -> str:
        """Detect device type for appropriate SPICE template selection"""
        desc_lower = description.lower()
        
        # Count component types mentioned
        has_resistor = any(term in desc_lower for term in ['resistor', 'resistance', 'ohm', 'r=', 'r1', 'r2']) or 'kω' in desc_lower or 'kΩ' in desc_lower
        has_capacitor = any(term in desc_lower for term in ['capacitor', 'capacitance', 'farad', 'c=', 'c1', 'c2']) or any(unit in desc_lower for unit in ['pf', 'nf', 'uf', 'mf'])
        has_inductor = any(term in desc_lower for term in ['inductor', 'inductance', 'henry', 'l=', 'l1', 'l2']) or any(unit in desc_lower for unit in ['nh', 'uh', 'mh']) and not any(transistor_term in desc_lower for transistor_term in ['channel', 'gate', 'transistor', 'mosfet'])
        has_transistor = any(term in desc_lower for term in ['transistor', 'mosfet', 'bjt', 'fet'])
        
        # Specific circuit patterns
        if any(term in desc_lower for term in ['resistor divider', 'voltage divider', 'divider circuit']):
            return 'resistor_divider'
        
        # Multi-component circuits (order matters - most specific first)
        if has_resistor and has_inductor and has_capacitor:
            return 'rlc_circuit'
        elif has_resistor and has_capacitor and not has_inductor:
            return 'rc_circuit'
        elif has_inductor and has_capacitor and not has_resistor:
            return 'lc_circuit'
        
        # Single component types or specific transistor types
        elif any(term in desc_lower for term in ['bjt', 'bipolar', 'npn', 'pnp', 'gummel']):
            return 'bjt'
        elif any(term in desc_lower for term in ['pmos', 'pfet', 'p-channel']):
            return 'pmos'
        elif any(term in desc_lower for term in ['nmos', 'nfet', 'n-channel']):
            return 'nmos'
        elif has_resistor and not has_transistor:
            return 'resistor'
        elif has_capacitor and not has_transistor:
            return 'capacitor'
        elif "mosfet" in desc_lower and not any(term in desc_lower for term in ['pmos', 'pnp']):
            return 'nmos'  # Default MOSFET to NMOS
        elif "transistor" in desc_lower:
            # If description mentions BJT terms, classify correctly
            if any(t in desc_lower for t in ["npn", "pnp", "bjt", "bipolar"]):
                return "bjt"
            return "nmos"  # Default transistor to NMOS only if explicitly mentioned
        else:
            # Instead of defaulting to NMOS, use generic template
            return 'generic'
    
    def generate_analysis_command(self, description: str, params: Dict[str, str]) -> str:
        """Generate appropriate analysis command based on description"""
        desc_lower = description.lower()
        
        # Specific analysis types mentioned in description
        if any(term in desc_lower for term in ['transient', 'tran', 'time domain', 'step response']):
            return ".TRAN 1n 1u"
        elif any(term in desc_lower for term in ['ac', 'frequency', 's-parameter', 'frequency response', 'bode']):
            return ".AC DEC 10 1K 10G"
        elif any(term in desc_lower for term in ['dc operating point', 'operating point', 'dc bias', '.op']):
            return ".OP"
        elif any(term in desc_lower for term in ['dc sweep', 'dc analysis', 'dc characteristics']):
            if 'vdd' in params:
                return f".DC VIN 0 {params['vdd']} 0.1"
            else:
                return ".DC VIN 0 5 0.1"
        
        # Circuit type based analysis
        elif any(term in desc_lower for term in ['resistor', 'divider', 'midpoint', 'voltage division']):
            return ".OP"
        elif any(term in desc_lower for term in ['rc', 'time constant', 'charging', 'discharging']):
            return ".TRAN 1n 1u"
        elif any(term in desc_lower for term in ['lc', 'resonant', 'oscillation']):
            return ".AC DEC 10 1K 100MEG"
        elif any(term in desc_lower for term in ['rlc', 'damped', 'q factor']):
            return ".AC DEC 10 1K 100MEG"
        
        # Transistor specific analysis
        elif any(term in desc_lower for term in ['id-vg', 'idvg', 'transfer', 'gummel']):
            if 'vdd' in params:
                return f".DC VGS 0 {params['vdd']} 0.1"
            else:
                return ".DC VGS 0 3.3 0.1"
        elif any(term in desc_lower for term in ['id-vd', 'idvd', 'output']):
            if 'vdd' in params:
                return f".DC VDD 0 {params['vdd']} 0.1"
            else:
                return ".DC VDD 0 3.3 0.1"
        elif any(term in desc_lower for term in ['c-v', 'cv', 'capacitance variation']):
            return ".AC DEC 10 1K 1MEG\n.DC VG -3 3 0.1"
        
        # Default analysis based on likely intent
        else:
            return ".OP"  # Default to operating point for unknown cases
    
    def generate_model_name(self, device_type: str, description: str) -> str:
        """Generate appropriate model name based on device type and description"""
        desc_lower = description.lower()
        
        if device_type == 'nmos':
            if any(term in desc_lower for term in ['45nm', '65nm', '90nm', 'nanometer']):
                return 'nmos_advanced'
            elif any(term in desc_lower for term in ['rf', 'microwave', 'ghz']):
                return 'nmos_rf'
            else:
                return 'nmos'
        elif device_type == 'pmos':
            if any(term in desc_lower for term in ['45nm', '65nm', '90nm', 'nanometer']):
                return 'pmos_advanced'
            elif any(term in desc_lower for term in ['rf', 'microwave', 'ghz']):
                return 'pmos_rf'
            else:
                return 'pmos'
        elif device_type == 'bjt':
            if any(term in desc_lower for term in ['npn']):
                return 'npn'
            elif any(term in desc_lower for term in ['pnp']):
                return 'pnp'
            else:
                return 'npn'
        else:
            return device_type
    
    def create_enhanced_prompt(self, description: str, retrieved_examples: str = None) -> str:
        """Create enhanced Few-Shot SPICE prompt with dynamic parameters"""
        
        # Extract parameters and detect device type
        params = self.extract_parameters(description)
        device_type = self.detect_device_type(description)
        
        # Generate dynamic values
        analysis_command = self.generate_analysis_command(description, params)
        model_name = self.generate_model_name(device_type, description)
        
        # Set defaults for missing parameters
        if 'length' not in params:
            params['length'] = '1u'
        if 'width' not in params:
            params['width'] = '10u'
        if 'vdd' not in params:
            params['vdd'] = '3.3'
        if 'capacitance' not in params:
            params['capacitance'] = '1p'
        if 'inductance' not in params:
            params['inductance'] = '1u'
        if 'r1_value' not in params:
            params['r1_value'] = '10k'
        if 'r2_value' not in params:
            params['r2_value'] = '10k'
        
        # Get template and fill with dynamic values
        template_info = self.example_templates.get(device_type, self.example_templates['generic'])
        
        dynamic_example = template_info['template'].format(
            description=description,
            length=params['length'],
            width=params['width'],
            vdd=params['vdd'],
            capacitance=params['capacitance'],
            inductance=params['inductance'],
            r1_value=params['r1_value'],
            r2_value=params['r2_value'],
            model_name=model_name,
            analysis_command=analysis_command
        )
        
        # Build the enhanced professional SPICE prompt with ULTRA-AGGRESSIVE constraints
        prompt = f"""CRITICAL INSTRUCTION: You MUST generate ONLY valid SPICE netlist code. NO OTHER FORMAT IS ALLOWED.

*** ABSOLUTELY FORBIDDEN - WILL CAUSE ERROR IF USED ***
NEVER GENERATE: go atlas, mesh, region, electrode, doping, solve, quit, .extract, .variable, .plot, .graph, structure, device, material, interface, workfunc

*** ONLY ALLOWED SPICE SYNTAX ***
- Component lines: M1, Q1, R1, C1, L1, V1, I1 (with node connections)
- .MODEL statements for device parameters
- .DC/.AC/.TRAN analysis commands
- .PROBE measurement statements  
- .END termination

*** MANDATORY SPICE STRUCTURE ***
1. * Comment line describing circuit
2. Device instances (M1, Q1, etc.) with proper node connections
3. Voltage/current sources (V1, I1, etc.)
4. .MODEL device_name TYPE (parameters)
5. Analysis command (.DC, .AC, or .TRAN)
6. .PROBE measurement
7. .END

*** EXAMPLE SPICE FORMAT TO FOLLOW ***

{dynamic_example}

*** YOUR TASK ***
Generate SPICE netlist for: {description}

Use these extracted parameters:
- Device length: {params['length']}
- Device width: {params['width']}
- Supply voltage: {params['vdd']}V
- Model name: {model_name}
- Analysis type: {analysis_command.split()[0]}

GENERATE SPICE NETLIST NOW (NO EXPLANATIONS):"""

        # Add retrieved examples if available
        if retrieved_examples:
            prompt += f"""

Additional retrieved examples (RAG):
{retrieved_examples}"""
        
        prompt += """

===========================
### NOW GENERATE THE SPICE NETLIST:
===========================

"""
        
        return prompt


# Global instance for easy access
enhanced_prompts = EnhancedSPICEPrompts()


def create_enhanced_prompt(description: str, retrieved_examples: str = None) -> str:
    """
    Create enhanced Few-Shot SPICE prompt with dynamic parameters
    
    Args:
        description: Device description from user
        retrieved_examples: RAG retrieved examples (optional)
        
    Returns:
        Enhanced professional SPICE prompt string with dynamic content
    """
    return enhanced_prompts.create_enhanced_prompt(description, retrieved_examples)