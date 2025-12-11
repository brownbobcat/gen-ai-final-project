#!/usr/bin/env python3
"""
rag_retrieve.py - Build and use FAISS index for SPICE example retrieval
Embeds .in netlists and provides similarity search functionality
"""

import os
import json
from pathlib import Path
from typing import List, Dict
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import re


class SpiceRAG:
    """RAG system for retrieving similar SPICE netlist examples"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with sentence transformer model"""
        print(f"Loading embedding model: {model_name}")
        self.embed_model = SentenceTransformer(model_name)
        self.index = None
        self.metadata = []
        self.embeddings = []
        
    def extract_key_features(self, code: str) -> str:
        """Extract key features from SPICE code for better embedding"""
        features = []
        
        # Extract device type from SPICE components
        device_patterns = [
            r'(MOSFET|mosfet|NMOS|PMOS|nmos|pmos)',
            r'(BJT|bjt|NPN|PNP|npn|pnp)',
            r'(DIODE|diode|Schottky|schottky)',
            r'(LNA|lna|amplifier|Amplifier)',
            r'(VCO|vco|oscillator|Oscillator)',
            r'(MIXER|mixer|Mixer)',
            r'(capacitor|inductor|resistor|Capacitor|Inductor|Resistor)'
        ]
        
        for pattern in device_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                features.append(re.search(pattern, code, re.IGNORECASE).group(1))
        
        # Extract SPICE component instances
        spice_components = [
            r'^(M\w+)',  # MOSFET
            r'^(Q\w+)',  # BJT
            r'^(D\w+)',  # Diode
            r'^(R\w+)',  # Resistor
            r'^(C\w+)',  # Capacitor
            r'^(L\w+)',  # Inductor
            r'^(V\w+)',  # Voltage source
            r'^(I\w+)',  # Current source
        ]
        
        for line in code.split('\n'):
            line = line.strip()
            for pattern in spice_components:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    features.append(match.group(1))
        
        # Extract SPICE parameters  
        param_patterns = [
            r'L=([0-9.]+[unm]?)',     # Length
            r'W=([0-9.]+[unm]?)',     # Width  
            r'DC\s+([0-9.]+)',        # DC voltage/current
            r'AC\s+([0-9.]+)',        # AC magnitude
            r'TEMP=([0-9.]+)',        # Temperature
            r'FUND=([0-9.]+[GMK]?Hz)' # Frequency
        ]
        
        for pattern in param_patterns:
            matches = re.findall(pattern, code, re.IGNORECASE)
            if matches:
                features.extend(matches[:3])  # Limit to first 3 matches
        
        # Extract simulation type
        sim_patterns = [
            r'\.(TRAN|tran)',
            r'\.(AC|ac)',
            r'\.(DC|dc)',
            r'\.(HARM|harm|HARMONIC)',
            r'\.(NOISE|noise)',
            r'\.(PSS|pss)',
            r'\.(HNET|hnet)',
            r'\.(HTF|htf)'
        ]
        
        for pattern in sim_patterns:
            if re.search(pattern, code):
                features.append(re.search(pattern, code).group(1).upper())
        
        # Combine features with first few lines of code
        code_lines = code.split('\n')[:10]  # First 10 lines
        feature_text = ' '.join(features) + ' ' + ' '.join(code_lines)
        
        return feature_text
    
    ATLAS_KEYWORDS = [
        'go atlas', 'mesh', 'region', 'electrode', 'doping', 'solve', 'plot',
        'structure', 'material', 'interface', 'method', 'workfunc', 'tonyplot'
    ]

    SPICE_REQUIRED_TOKENS = [
        '.model', '.subckt', '.include', '.options', '.tran', '.ac', '.dc'
    ]

    MIN_SPICE_LINES = 8

    def _is_valid_spice(self, content: str) -> bool:
        """Return True if content looks like SPICE and not ATLAS"""
        lower = content.lower()

        if any(keyword in lower for keyword in self.ATLAS_KEYWORDS):
            return False

        if not any(token in lower for token in self.SPICE_REQUIRED_TOKENS):
            # allow pure element-level netlists as long as they contain common components
            component_patterns = [r'^[mqrstclvi]\w+\s+', r'\.end\s*$', r'\.global']
            if not any(re.search(pattern, content, re.IGNORECASE) for pattern in component_patterns):
                return False

        # Require at least one component OR subcircuit definition
        has_component = any(
            re.match(r'^[MQRCLVI]\w+\s+', line.strip(), re.IGNORECASE)
            for line in content.splitlines()
        )
        has_subckt = '.subckt' in lower
        has_analysis = any(cmd in lower for cmd in ['.tran', '.ac', '.dc', '.noise', '.measure', '.op'])

        if not (has_component or has_subckt):
            return False

        # Require at least MIN_SPICE_LINES to avoid trivial snippets
        non_empty_lines = [line for line in content.splitlines() if line.strip()]
        if len(non_empty_lines) < self.MIN_SPICE_LINES:
            return False

        # Require at least one analysis or control statement to ensure runnable decks
        if not has_analysis and '.model' not in lower and '.include' not in lower:
            return False

        return True

    def load_spice_files(self, base_dir: str) -> List[Dict]:
        """Load SPICE-only .in files from directory, filtering out ATLAS decks"""
        files_data = []
        
        print(f"Scanning for .in files in {base_dir}")
        for root, _, files in os.walk(base_dir):
            for file in files:
                if not file.endswith('.in'):
                    continue

                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
                    continue

                if not self._is_valid_spice(content):
                    continue

                rel_path = os.path.relpath(filepath, base_dir)
                files_data.append({
                    'filepath': filepath,
                    'relative_path': rel_path,
                    'filename': file,
                    'content': content,
                    'length': len(content)
                })

        print(f"Found {len(files_data)} SPICE netlists")
        return files_data
    
    def build_index(self, examples_dir: str, index_path: str, metadata_path: str):
        """Build FAISS index from all .in files"""
        # Load all files
        files_data = self.load_spice_files(examples_dir)
        
        if not files_data:
            raise ValueError(f"No .in files found in {examples_dir}")
        
        # Create embeddings
        print("Creating embeddings...")
        texts_to_embed = []
        
        for file_data in tqdm(files_data, desc="Preparing texts"):
            # Extract features and create embedding text
            feature_text = self.extract_key_features(file_data['content'])
            texts_to_embed.append(feature_text)
        
        # Batch encode for efficiency
        print("Encoding texts to embeddings...")
        embeddings = self.embed_model.encode(
            texts_to_embed,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Normalize embeddings for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Build FAISS index
        print("Building FAISS index...")
        dimension = embeddings.shape[1]
        
        # Use IndexFlatIP for inner product (equivalent to cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata
        self.metadata = files_data
        self.embeddings = embeddings
        
        # Save index and metadata
        print(f"Saving index to {index_path}")
        faiss.write_index(self.index, index_path)
        
        print(f"Saving metadata to {metadata_path}")
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save some statistics
        stats = {
            'total_files': len(files_data),
            'embedding_dimension': dimension,
            'index_type': 'IndexFlatIP',
            'model_name': 'all-MiniLM-L6-v2',
            'average_file_length': np.mean([f['length'] for f in files_data]),
            'files_by_directory': {}
        }
        
        # Count files by directory
        for file_data in files_data:
            dir_name = os.path.dirname(file_data['relative_path']).split('/')[0]
            stats['files_by_directory'][dir_name] = stats['files_by_directory'].get(dir_name, 0) + 1
        
        stats_path = os.path.join(os.path.dirname(index_path), 'index_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\nIndex built successfully!")
        print(f"Total files indexed: {stats['total_files']}")
        print(f"Embedding dimension: {stats['embedding_dimension']}")
        
    def load_index(self, index_path: str, metadata_path: str):
        """Load existing FAISS index and metadata"""
        print(f"Loading index from {index_path}")
        self.index = faiss.read_index(index_path)
        
        print(f"Loading metadata from {metadata_path}")
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        print(f"Index loaded with {len(self.metadata)} examples")
    
    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve k most similar examples"""
        if self.index is None:
            raise ValueError("Index not loaded. Call load_index() first.")
        
        # Extract features from query
        query_features = self.extract_key_features(query)
        
        # Encode query
        query_embedding = self.embed_model.encode([query_features], convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Prepare results
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['similarity_score'] = float(score)
                results.append(result)
        
        return results
    
    def _extract_comment_summary(self, content: str, max_lines: int = 6) -> List[str]:
        """Return a few descriptive comment lines instead of raw code"""
        summary = []
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(('*', '#', ';', '!')):
                cleaned = stripped.lstrip('*#;! ').strip()
                if cleaned:
                    summary.append(cleaned)
            if len(summary) >= max_lines:
                break
        return summary
    
    def _extract_analysis_commands(self, content: str) -> List[str]:
        """List unique analysis commands present in the file"""
        analysis_patterns = {
            'AC analysis': r'\.ac\b',
            'Transient analysis': r'\.tran\b',
            'DC sweep': r'\.dc\b',
            'Harmonic balance': r'\.(hb|harm|harmonic)\b',
            'Noise analysis': r'\.noise\b',
            'Parameter sweep': r'\.sweep\b',
            'Extraction/logging': r'(log|extract)'
        }
        found = []
        content_lower = content.lower()
        for label, pattern in analysis_patterns.items():
            if re.search(pattern, content_lower, re.IGNORECASE):
                found.append(label)
        return found
    
    def _extract_key_parameters(self, content: str, max_params: int = 5) -> List[str]:
        """Get a few parameter/value strings to hint at scales without dumping code"""
        patterns = [
            r'(\bL\s*=\s*[0-9.]+[unm]?)',
            r'(\bW\s*=\s*[0-9.]+[unm]?)',
            r'(vdd\s*=\s*[0-9.]+[vV]?)',
            r'(freq(?:uency)?\s*=\s*[0-9.]+[GMk]?Hz)',
            r'(thickness\s*=\s*[0-9.]+[unm]?)',
            r'(doping\s*=\s*[0-9.e+-]+)'
        ]
        params = []
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if match not in params:
                    params.append(match.strip())
                if len(params) >= max_params:
                    return params
        return params
    
    def format_retrieved_examples(self, examples: List[Dict]) -> str:
        """Format retrieved examples for SPICE few-shot integration"""
        if not examples:
            return ""

        formatted = []

        for i, example in enumerate(examples[:2]):
            content = example['content']
            content_lower = content.lower()

            # Skip any files that look like TCAD/ATLAS decks
            if any(term in content_lower for term in ['go atlas', 'mesh', 'region', 'electrode']):
                continue

            formatted_example = f"\n--- RAG Retrieved SPICE Example {i+1} ---\n"
            formatted_example += f"Source: {example['relative_path']}\n"
            formatted_example += f"Similarity Score: {example.get('similarity_score', 0.0):.3f}\n"

            comment_summary = self._extract_comment_summary(content)
            if comment_summary:
                formatted_example += "Summary: " + '; '.join(comment_summary[:2]) + "\n"

            key_sections = []
            for line in content.split('\n'):
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith('*') and len(key_sections) < 2:
                    key_sections.append(stripped)
                    continue

                if stripped[0:1].upper() in ['M', 'Q', 'R', 'C', 'L', 'V', 'I', 'X'] or \
                   any(token in stripped.lower() for token in ['.model', '.subckt', '.dc', '.ac', '.tran', '.end', '.measure']):
                    key_sections.append(stripped)
                if len(key_sections) >= 18:
                    break

            if not any('.model' in line.lower() for line in key_sections):
                continue

            formatted_example += '\n'.join(key_sections)
            formatted_example += "\n" + "=" * 50
            formatted.append(formatted_example)

        return '\n'.join(formatted)


def main():
    """Build the FAISS index"""
    root_dir = Path(__file__).resolve().parents[1]
    examples_dir = root_dir / "data" / "SilvacoUserManuel_CodeExamples" / "examples"
    embeddings_dir = root_dir / "embeddings"
    index_path = embeddings_dir / "faiss_index.bin"
    metadata_path = embeddings_dir / "metadata.json"
    
    # Create embeddings directory
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Initialize RAG system
    rag = SpiceRAG()
    
    # Build index
    rag.build_index(str(examples_dir), str(index_path), str(metadata_path))
    
    # Test retrieval
    print("\n" + "="*50)
    print("Testing retrieval with sample query...")
    test_query = "Create a MOSFET transistor with 2V supply voltage and harmonic balance analysis"
    
    rag.load_index(str(index_path), str(metadata_path))
    results = rag.retrieve(test_query, k=3)
    
    print(f"\nQuery: {test_query}")
    print(f"\nTop 3 retrieved examples:")
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result['relative_path']}")
        print(f"   Similarity: {result['similarity_score']:.3f}")
        print(f"   First 200 chars: {result['content'][:200]}...")


if __name__ == "__main__":
    main()
