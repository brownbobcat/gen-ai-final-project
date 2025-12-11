#!/usr/bin/env python3
"""
Dynamic SPICE template system for netlist generation
"""

import re

class SPICETemplateEngine:
    """Dynamic SPICE template generation for any device type"""
    
    def __init__(self):
        self.device_configs = {
            'nmos': {
                'component': 'M',
                'model_type': 'NMOS',
                'default_vto': '0.7',
                'default_kp': '120u',
                'default_vdd': '3.3',
                'default_analysis': '.DC VGS 0 3.3 0.1',
            },
            'pmos': {
                'component': 'M',
                'model_type': 'PMOS',
                'default_vto': '-0.7',
                'default_kp': '50u',
                'default_vdd': '3.3',
                'default_analysis': '.DC VGS 0 -3.3 -0.1',
            },
            'npn': {
                'component': 'Q',
                'model_type': 'NPN',
                'default_is': '1e-16',
                'default_bf': '100',
                'default_vcc': '5.0',
                'default_analysis': '.DC VBE 0.4 1.0 0.01',
            },
            'pnp': {
                'component': 'Q', 
                'model_type': 'PNP',
                'default_is': '1e-16',
                'default_bf': '100',
                'default_vcc': '5.0',
                'default_analysis': '.DC VBE -0.4 -1.0 -0.01',
            },
            'bjt': {'alias': 'npn'},
            'mosfet': {'alias': 'nmos'},
            'transistor': {'alias': 'nmos'},  # default
        }
    
    def parse_parameters(self, text):
        """Extract all SPICE parameters from description"""
        params = {}
        
        # Channel length patterns for MOSFETs
        l_patterns = [
            r'L\s*=\s*(\d+\.?\d*)\s*([μµun]m?)',
            r'(?:channel|gate).*?length.*?(\d+\.?\d*)\s*([μµun]m?)',
            r'(\d+)\s*nm.*?(?:channel|gate)',
        ]
        
        for pattern in l_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                unit = match.group(2).lower() if len(match.groups()) > 1 else ''
                if 'n' in unit or 'nm' in text.lower():
                    params['L'] = f"{int(value) if value == int(value) else value}n"
                else:
                    params['L'] = f"{int(value) if value == int(value) else value}u"
                break
        
        # Channel width patterns for MOSFETs
        w_patterns = [
            r'W\s*=\s*(\d+\.?\d*)\s*([μµun]m?)', 
            r'(?:channel|gate).*?width.*?(\d+\.?\d*)\s*([μµun]m?)',
            r'(\d+\.?\d*)\s*([μµun]m?).*?width'
        ]
        
        for pattern in w_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                unit = match.group(2).lower() if len(match.groups()) > 1 else ''
                if 'n' in unit:
                    params['W'] = f"{int(value) if value == int(value) else value}n"
                else:
                    params['W'] = f"{int(value) if value == int(value) else value}u"
                break
        
        # Voltage patterns
        v_patterns = [
            r'(\d+\.?\d*)\s*V\s+(?:supply|voltage|VDD)',
            r'VDD\s*=\s*(\d+\.?\d*)\s*V?',
            r'supply.*?(\d+\.?\d*)\s*V',
        ]
        
        for pattern in v_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                params['VDD'] = f"{value}" if value != int(value) else f"{int(value)}"
                break
        
        return params
    
    def detect_device_type(self, text):
        """Detect device type from description"""
        text_lower = text.lower()
        
        if any(term in text_lower for term in ['pmos', 'pfet', 'p-channel']):
            return 'pmos'
        elif any(term in text_lower for term in ['bjt', 'bipolar', 'npn']):
            return 'npn'
        elif any(term in text_lower for term in ['pnp']):
            return 'pnp'
        elif any(term in text_lower for term in ['nmos', 'nfet', 'n-channel', 'mosfet']):
            return 'nmos'
        elif "transistor" in text_lower:
            if any(t in text_lower for t in ["npn", "pnp", "bjt", "bipolar"]):
                return 'npn'
            return 'nmos'
        else:
            return 'nmos'  # Default
    
    def detect_analysis_type(self, text, device_type):
        """Detect required analysis from description"""
        text_lower = text.lower()

        if device_type in ["npn", "pnp"] and "transfer" in text_lower:
            return "dc_transfer_bjt"
        
        if any(term in text_lower for term in ['id-vg', 'idvg', 'transfer', 'gummel']):
            return 'dc_transfer'
        elif any(term in text_lower for term in ['id-vd', 'idvd', 'output']):
            return 'dc_output'
        elif any(term in text_lower for term in ['c-v', 'cv', 'capacitance']):
            return 'ac_cv'
        elif any(term in text_lower for term in ['ac', 'frequency', 's-parameter']):
            return 'ac'
        elif any(term in text_lower for term in ['transient', 'tran', 'time']):
            return 'transient'
        else:
            return 'dc_transfer'  # Default
    
    def generate_analysis_command(self, device_type, analysis_type, params):
        """Generate appropriate analysis command"""
        vdd = params.get('VDD', '3.3')
        
        if analysis_type == 'dc_transfer_bjt':
            if device_type == "npn":
                return ".DC VBE 0.4 1.0 0.01"
            else:
                return ".DC VBE -0.4 -1.0 -0.01"
        if analysis_type == 'dc_transfer':
            if device_type in ['nmos', 'npn']:
                return f".DC VGS 0 {vdd} 0.1" if device_type == 'nmos' else f".DC VBE 0.4 1.0 0.01"
            else:  # pmos, pnp
                return f".DC VGS 0 -{vdd} -0.1" if device_type == 'pmos' else f".DC VBE -0.4 -1.0 -0.01"
        elif analysis_type == 'dc_output':
            if device_type in ['nmos', 'npn']:
                return f".DC VDD 0 {vdd} 0.1"
            else:
                return f".DC VDD 0 -{vdd} -0.1"
        elif analysis_type == 'ac_cv':
            return ".AC DEC 10 1K 1MEG\n.DC VG -3 3 0.1"
        elif analysis_type == 'ac':
            return ".AC DEC 10 1K 10G"
        elif analysis_type == 'transient':
            return ".TRAN 1n 1u"
        else:
            return f".DC VGS 0 {vdd} 0.1"
    
    def generate_complete_netlist(self, description):
        """Generate complete SPICE netlist from description"""
        
        # Parse description
        device_type = self.detect_device_type(description)
        analysis_type = self.detect_analysis_type(description, device_type)
        params = self.parse_parameters(description)
        
        # Get device config
        config = self.device_configs.get(device_type, self.device_configs['nmos'])
        if 'alias' in config:
            config = self.device_configs[config['alias']]
        
        # Set defaults
        length = params.get('L', '1u')
        width = params.get('W', '10u')
        vdd = params.get('VDD', config.get('default_vdd', '3.3'))
        
        # Generate netlist
        netlist = f"* {description}\n"
        
        if config['component'] == 'M':  # MOSFET
            model_name = f"{device_type}_model"
            netlist += f"M1 vd vg 0 0 {model_name} L={length} W={width}\n"
            netlist += f"VDD vd 0 DC {vdd}\n"
            netlist += f"VGS vg 0 DC 0\n"
            netlist += f"VSS 0 0 DC 0\n\n"
            
            # Add model
            vto = config.get('default_vto', '0.7')
            kp = config.get('default_kp', '120u')
            netlist += f".MODEL {model_name} {config['model_type']}(\n"
            netlist += f"+ LEVEL=1 VTO={vto} KP={kp} GAMMA=0.5 PHI=0.6\n"
            netlist += f"+ LAMBDA=0.1 RD=10 RS=10 CBD=5f CBS=5f\n"
            netlist += f"+ CGDO=0.4n CGSO=0.4n CGBO=0.4n)\n\n"
            
        elif config['component'] == 'Q':  # BJT
            model_name = f"{device_type}_model"
            netlist += f"Q1 vc vb 0 {model_name}\n"
            netlist += f"VCC vc 0 DC {config.get('default_vcc', '5.0')}\n"
            netlist += f"VBE vb 0 DC 0\n\n"
            
            # Add model
            is_val = config.get('default_is', '1e-16')
            bf = config.get('default_bf', '100')
            netlist += f".MODEL {model_name} {config['model_type']}(\n"
            netlist += f"+ IS={is_val} BF={bf} BR=1 VAF=100 VAR=10\n"
            netlist += f"+ RB=10 RC=1 RE=0.5 CJE=0.5p CJC=0.3p)\n\n"
        
        # Add analysis
        analysis_cmd = self.generate_analysis_command(device_type, analysis_type, params)
        netlist += analysis_cmd + "\n"
        
        # Add probe
        if config['component'] == 'M':
            netlist += ".PROBE DC I(VDD)\n"
        else:
            netlist += ".PROBE DC I(VCC) I(VBE)\n"
        
        return netlist.strip() + "\n.END\n"

# Global instance for easy access
template_engine = SPICETemplateEngine()
