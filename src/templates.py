#!/usr/bin/env python3
"""
Dynamic template system for Silvaco device generation
"""

import re

class SilvacoTemplateEngine:
    """Dynamic template generation for any device type"""
    
    def __init__(self):
        self.device_configs = {
            'nmos': {
                'substrate': 'p.type',
                'substrate_conc': '1e15',
                'source_drain_type': 'n.type',
                'default_vd': 0.1,
                'default_vg_range': (-1.0, 3.0),
                'default_vd_range': (0.0, 3.0),
            },
            'pmos': {
                'substrate': 'n.type',
                'substrate_conc': '1e15', 
                'source_drain_type': 'p.type',
                'default_vd': -0.1,
                'default_vg_range': (1.0, -3.0),
                'default_vd_range': (0.0, -3.0),
            },
            'nfet': {'alias': 'nmos'},
            'pfet': {'alias': 'pmos'},
            'transistor': {'alias': 'nmos'},  # default
        }
    
    def parse_dimensions(self, text):
        """Extract all dimensions from description"""
        dims = {}
        
        # Channel length patterns
        l_patterns = [
            r'L\s*=\s*(\d+\.?\d*)\s*([μµun]m?)',
            r'(?:channel|gate).*?length.*?(\d+\.?\d*)\s*([μµun]m?)',
            r'(\d+\.?\d*)\s*([μµun]m?).*?(?:channel|gate).*?length'
        ]
        
        # Channel width patterns  
        w_patterns = [
            r'W\s*=\s*(\d+\.?\d*)\s*([μµun]m?)', 
            r'(?:channel|gate).*?width.*?(\d+\.?\d*)\s*([μµun]m?)',
            r'(\d+\.?\d*)\s*([μµun]m?).*?(?:channel|gate).*?width'
        ]
        
        # Oxide thickness patterns
        tox_patterns = [
            r'(?:oxide|gate).*?thickness.*?(\d+\.?\d*)\s*([μµun]m?)',
            r'tox\s*=?\s*(\d+\.?\d*)\s*([μµun]m?)',
            r'(\d+\.?\d*)\s*([μµun]m?).*?oxide.*?thick'
        ]
        
        # Extract dimensions
        for patterns, key in [(l_patterns, 'L'), (w_patterns, 'W'), (tox_patterns, 'tox')]:
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    value = float(match.group(1))
                    unit = match.group(2).lower()
                    
                    # Convert to micrometers
                    if 'n' in unit:  # nm
                        value_um = value / 1000.0
                        dims[key] = f"{value}n"
                    else:  # um or just m
                        value_um = value
                        dims[key] = f"{value}u"
                    
                    dims[f"{key}_um"] = value_um
                    break
        
        return dims
    
    def parse_doping(self, text):
        """Extract doping concentrations"""
        doping = {}
        
        patterns = [
            r'(\d+\.?\d*)\s*[×x]\s*10\^?(\d+).*?cm\^?-?3',
            r'10\^?(\d+).*?cm\^?-?3', 
            r'1e(\d+).*?cm\^?-?3',
            r'(\d+\.?\d*)\s*e(\d+).*?cm\^?-?3'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) == 2 and ('x' in pattern or '×' in pattern or 'e' in pattern):
                    mantissa = float(groups[0])
                    exponent = int(groups[1])
                    doping['source_drain'] = f"{mantissa}e{exponent}"
                else:
                    exponent = int(groups[0])
                    doping['source_drain'] = f"1e{exponent}"
                break
        
        return doping
    
    def parse_analysis(self, text):
        """Determine analysis type from description"""
        text_lower = text.lower()
        
        if any(term in text_lower for term in ['id-vd', 'idvd', 'output', 'drain']):
            return 'id_vd'
        elif any(term in text_lower for term in ['id-vg', 'idvg', 'transfer', 'gate']):
            return 'id_vg'
        elif any(term in text_lower for term in ['cv', 'c-v', 'capacitance']):
            return 'cv'
        elif any(term in text_lower for term in ['ac', 'frequency']):
            return 'ac'
        
        return 'id_vg'  # default
    
    def detect_device_type(self, text):
        """Detect device type from description"""
        text_lower = text.lower()
        
        # Direct device type detection
        if any(term in text_lower for term in ['pmos', 'pfet', 'p-channel', 'p channel']):
            return 'pmos'
        elif any(term in text_lower for term in ['nmos', 'nfet', 'n-channel', 'n channel']):
            return 'nmos'
        elif any(term in text_lower for term in ['diode', 'pn', 'p-n']):
            return 'diode'
        elif any(term in text_lower for term in ['bjt', 'bipolar', 'npn', 'pnp']):
            return 'bjt'
        elif any(term in text_lower for term in ['jfet', 'junction fet']):
            return 'jfet'
        elif any(term in text_lower for term in ['mosfet', 'fet', 'transistor']):
            return 'nmos'  # default to NMOS
        
        return 'nmos'  # ultimate default
    
    def generate_mesh(self, dims, device_type='nmos'):
        """Generate adaptive mesh based on device dimensions"""
        L = dims.get('L_um', 1.0)
        W = dims.get('W_um', 10.0)
        
        # Adaptive mesh spacing
        x_fine = min(0.002, L/500)
        x_coarse = min(0.01, L/100)
        y_fine = 0.001
        
        # Mesh extends beyond device
        x_ext = max(0.5, L/2)
        
        mesh = f"""# Adaptive mesh for L={dims.get('L', '1u')}, W={dims.get('W', '10u')} {device_type.upper()}
mesh space.mult=1.0
x.mesh l={-x_ext:.3f} spac={x_coarse:.4f}
x.mesh l=0.0 spac={x_fine:.5f}
x.mesh l={L:.3f} spac={x_fine:.5f}
x.mesh l={L + x_ext:.3f} spac={x_coarse:.4f}
y.mesh l=0.0 spac={y_fine:.4f}"""

        # Add oxide mesh if specified
        if 'tox_um' in dims:
            tox = dims['tox_um']
            mesh += f"\ny.mesh l={tox:.4f} spac={y_fine/2:.5f}"
            mesh += f"\ny.mesh l={tox + 0.01:.3f} spac=0.002"
        else:
            mesh += "\ny.mesh l=0.01 spac=0.002"
        
        mesh += "\ny.mesh l=0.1 spac=0.01\ny.mesh l=0.5 spac=0.05"
        
        return mesh
    
    def generate_regions(self, dims, device_type='nmos'):
        """Generate material regions"""
        L = dims.get('L_um', 1.0)
        x_ext = max(0.5, L/2)
        config = self.device_configs.get(device_type, self.device_configs['nmos'])
        
        regions = f"""# Material regions
region num=1 material=silicon x.min={-x_ext:.3f} x.max={L + x_ext:.3f} y.min=0.0 y.max=0.5"""

        # Add gate oxide if MOSFETs
        if device_type in ['nmos', 'pmos']:
            tox = dims.get('tox_um', 0.01)  # Default 10nm
            regions += f"""
region num=2 material=oxide x.min=0.0 x.max={L:.3f} y.min=0.0 y.max={tox:.4f}"""
        
        return regions
    
    def generate_electrodes(self, dims, device_type='nmos'):
        """Generate electrode definitions"""
        L = dims.get('L_um', 1.0)
        x_ext = max(0.5, L/2)
        tox = dims.get('tox_um', 0.01)
        
        electrodes = f"""# Electrodes
electrode num=1 name=source x.min={-x_ext:.3f} x.max=0.0 y.min=0.0 y.max=0.0
electrode num=2 name=drain x.min={L:.3f} x.max={L + x_ext:.3f} y.min=0.0 y.max=0.0"""

        if device_type in ['nmos', 'pmos']:
            electrodes += f"""
electrode num=3 name=gate x.min=0.0 x.max={L:.3f} y.min={tox:.4f} y.max={tox:.4f}"""
        elif device_type == 'bjt':
            electrodes += f"""
electrode num=3 name=base x.min={L/3:.3f} x.max={2*L/3:.3f} y.min=0.0 y.max=0.0"""
            
        return electrodes
    
    def generate_doping(self, dims, doping_params, device_type='nmos'):
        """Generate doping profiles"""
        L = dims.get('L_um', 1.0)
        x_ext = max(0.5, L/2)
        config = self.device_configs.get(device_type, self.device_configs['nmos'])
        
        substrate_type = config['substrate']
        substrate_conc = config['substrate_conc']
        sd_type = config['source_drain_type']
        sd_conc = doping_params.get('source_drain', '1e17')
        
        doping = f"""# Doping profiles
doping uniform {substrate_type} conc={substrate_conc} region=1

# Source/drain regions
doping gaussian {sd_type} conc={sd_conc} char.length=0.05 x.min={-x_ext:.3f} x.max=0.0
doping gaussian {sd_type} conc={sd_conc} char.length=0.05 x.min={L:.3f} x.max={L + x_ext:.3f}"""

        return doping
    
    def generate_analysis(self, dims, device_type, analysis_type='id_vg'):
        """Generate analysis commands"""
        config = self.device_configs.get(device_type, self.device_configs['nmos'])
        L = dims.get('L', '1u')
        
        analysis = """# Physical models
models srh auger fermi

# Initial solution
solve initial"""
        
        if analysis_type == 'id_vg':
            vd = config['default_vd']
            vg_start, vg_end = config['default_vg_range']
            analysis += f"""

# Id-Vg curve extraction
solve vdrain={vd} vgate={vg_start} initial
log outf=idvg_{device_type}_{L}.log
solve vgate={vg_end} vstep={0.1 if vg_end > vg_start else -0.1} name=gate
save outf={device_type}_{L}_idvg.str

tonyplot idvg_{device_type}_{L}.log"""

        elif analysis_type == 'id_vd':
            vg = 1.0 if device_type == 'nmos' else -1.0
            vd_start, vd_end = config['default_vd_range']
            analysis += f"""

# Id-Vd curve extraction
solve vgate={vg} vdrain={vd_start} initial  
log outf=idvd_{device_type}_{L}.log
solve vdrain={vd_end} vstep={0.1 if vd_end > vd_start else -0.1} name=drain
save outf={device_type}_{L}_idvd.str

tonyplot idvd_{device_type}_{L}.log"""

        elif analysis_type == 'cv':
            analysis += f"""

# C-V analysis
solve vgate=-2.0 vdrain=0.0 initial
log outf=cv_{device_type}_{L}.log
solve vgate=2.0 vstep=0.1 name=gate ac freq=1e6
save outf={device_type}_{L}_cv.str

tonyplot cv_{device_type}_{L}.log"""
        
        return analysis
    
    def generate_complete_deck(self, description):
        """Generate complete Silvaco deck from description"""
        
        # Parse all parameters
        device_type = self.detect_device_type(description)
        dims = self.parse_dimensions(description)
        doping_params = self.parse_doping(description)
        analysis_type = self.parse_analysis(description)
        
        # Handle device aliases
        config = self.device_configs.get(device_type, {})
        if 'alias' in config:
            device_type = config['alias']
        
        # Set defaults if not found
        if 'L' not in dims:
            dims.update({'L': '1u', 'L_um': 1.0})
        if 'W' not in dims:
            dims.update({'W': '10u', 'W_um': 10.0})
        
        # Generate complete deck
        deck = f"""go atlas

{self.generate_mesh(dims, device_type)}

{self.generate_regions(dims, device_type)}

{self.generate_electrodes(dims, device_type)}

{self.generate_doping(dims, doping_params, device_type)}

{self.generate_analysis(dims, device_type, analysis_type)}

quit
"""
        
        return deck.strip()


# Global instance for easy access
template_engine = SilvacoTemplateEngine()

def get_template(device_type, **params):
    """Legacy compatibility function"""
    # Convert old parameters to description format
    description = f"{device_type} transistor"
    if 'channel_length' in params:
        description += f" with {params['channel_length']} channel length"
    if 'doping' in params:
        description += f" and {params['doping']} source/drain doping"
    if params.get('analysis') == 'id_vg':
        description += ". Extract Id-Vg curve"
    elif params.get('analysis') == 'id_vd':
        description += ". Extract Id-Vd curve"
    
    return template_engine.generate_complete_deck(description)