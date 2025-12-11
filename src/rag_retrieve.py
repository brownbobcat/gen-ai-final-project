#!/usr/bin/env python3
"""
rag_retrieve.py - Build and use FAISS index for Silvaco example retrieval
Embeds all .in files and provides similarity search functionality
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import re


class SilvacoRAG:
    """RAG system for retrieving similar Silvaco examples"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with sentence transformer model"""
        print(f"Loading embedding model: {model_name}")
        self.embed_model = SentenceTransformer(model_name)
        self.index = None
        self.metadata = []
        self.embeddings = []
        
    def extract_key_features(self, code: str) -> str:
        """Extract key features from Silvaco code for better embedding"""
        features = []
        
        # Extract device type from comments
        device_patterns = [
            r'(MOSFET|mosfet|NMOS|PMOS|nmos|pmos)',
            r'(BJT|bjt|NPN|PNP|npn|pnp)',
            r'(DIODE|diode|Schottky|schottky)',
            r'(LNA|lna|amplifier|Amplifier)',
            r'(VCO|vco|oscillator|Oscillator)',
            r'(MIXER|mixer|Mixer)',
            r'(photonic|Photonic|optical|Optical)',
            r'(sensor|Sensor|detector|Detector)'
        ]
        
        for pattern in device_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                features.append(re.search(pattern, code, re.IGNORECASE).group(1))
        
        # Extract key parameters
        param_patterns = [
            r'\.PARAM\s+(\w+)',
            r'WIDTH=([0-9.]+[unm]?)',
            r'LENGTH=([0-9.]+[unm]?)',
            r'FUND=([0-9.]+[GMK]?Hz)',
            r'VDD.*?([0-9.]+)',
            r'TEMP.*?([0-9.]+)'
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
    
    def load_silvaco_files(self, base_dir: str) -> List[Dict]:
        """Load all .in files from directory"""
        files_data = []
        
        print(f"Scanning for .in files in {base_dir}")
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.in'):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # Extract relative path for better organization
                        rel_path = os.path.relpath(filepath, base_dir)
                        
                        files_data.append({
                            'filepath': filepath,
                            'relative_path': rel_path,
                            'filename': file,
                            'content': content,
                            'length': len(content)
                        })
                    except Exception as e:
                        print(f"Error reading {filepath}: {e}")
        
        print(f"Found {len(files_data)} .in files")
        return files_data
    
    def build_index(self, examples_dir: str, index_path: str, metadata_path: str):
        """Build FAISS index from all .in files"""
        # Load all files
        files_data = self.load_silvaco_files(examples_dir)
        
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
    
    def _extract_section_presence(self, content: str) -> List[str]:
        """Capture which Silvaco sections this file demonstrates"""
        section_keywords = [
            ('Mesh definition', 'mesh'),
            ('Region setup', 'region'),
            ('Electrodes/contacts', 'electrode'),
            ('Material blocks', 'material'),
            ('Doping profiles', 'doping'),
            ('Models section', 'models'),
            ('Solve commands', 'solve'),
            ('Atlas invocation', 'go atlas'),
            ('Quit statement', 'quit')
        ]
        content_lower = content.lower()
        sections = []
        for label, keyword in section_keywords:
            if keyword in content_lower:
                sections.append(label)
        return sections
    
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
        """Format retrieved examples as short natural-language summaries"""
        formatted = []
        
        for i, example in enumerate(examples):
            comments = self._extract_comment_summary(example['content'])
            sections = self._extract_section_presence(example['content'])
            analyses = self._extract_analysis_commands(example['content'])
            params = self._extract_key_parameters(example['content'])
            
            summary_lines = [
                f"Example {i+1}: {example['relative_path']}",
                "Use case summary: " + (comments[0] if comments else "Not specified"),
                "Key focuses: " + (', '.join(comments[1:3]) if len(comments) > 1 else "N/A"),
                "Silvaco sections demonstrated: " + (', '.join(sections) if sections else "unspecified"),
                "Analysis commands: " + (', '.join(analyses) if analyses else "none noted"),
                "Notable parameters: " + (', '.join(params) if params else "not provided")
            ]
            
            formatted.append('\n'.join(summary_lines))
        
        return "\n\n---\n\n".join(formatted)


def main():
    """Build the FAISS index"""
    # Paths
    examples_dir = "data/SilvacoUserManuel_CodeExamples/examples"
    index_path = "embeddings/faiss_index.bin"
    metadata_path = "embeddings/metadata.json"
    
    # Create embeddings directory
    os.makedirs("embeddings", exist_ok=True)
    
    # Initialize RAG system
    rag = SilvacoRAG()
    
    # Build index
    rag.build_index(examples_dir, index_path, metadata_path)
    
    # Test retrieval
    print("\n" + "="*50)
    print("Testing retrieval with sample query...")
    test_query = "Create a MOSFET transistor with 2V supply voltage and harmonic balance analysis"
    
    rag.load_index(index_path, metadata_path)
    results = rag.retrieve(test_query, k=3)
    
    print(f"\nQuery: {test_query}")
    print(f"\nTop 3 retrieved examples:")
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result['relative_path']}")
        print(f"   Similarity: {result['similarity_score']:.3f}")
        print(f"   First 200 chars: {result['content'][:200]}...")


if __name__ == "__main__":
    main()
