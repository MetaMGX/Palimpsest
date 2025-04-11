# palimpsest/src/main.py
"""
Main entry point for the Palimpsest application.

This module integrates the various analysis components and provides
interfaces for text analysis and comparison workflows.
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional
import argparse

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Palimpsest modules
from src.core.string_matching_module import StringMatchingModule
from src.core.semantic_analysis_module import SemanticAnalysisModule

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('palimpsest.log')
    ]
)
logger = logging.getLogger(__name__)

class PalimpsestAnalyzer:
    """
    Main class for the Palimpsest text analysis system.
    
    Integrates string matching and semantic analysis to find connections
    between texts at multiple levels.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Palimpsest analyzer with configuration.
        
        Args:
            config: Configuration dictionary for analyzer modules
        """
        self.config = config or {}
        logger.info("Initializing Palimpsest Analyzer")
        
        # Initialize analysis modules
        self.string_matcher = StringMatchingModule()
        self.semantic_analyzer = SemanticAnalysisModule()
        
        logger.info("Palimpsest Analyzer initialized successfully")
    
    def load_text(self, file_path: str) -> str:
        """
        Load text from a file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            The text content as a string
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            logger.info(f"Loaded text from {file_path}, length: {len(text)} characters")
            return text
        except Exception as e:
            logger.error(f"Error loading text from {file_path}: {e}")
            raise
    
    def analyze_texts(self, source_text: str, target_texts: List[str]) -> Dict[str, Any]:
        """
        Analyze relationships between a source text and multiple target texts.
        
        Args:
            source_text: Main text to analyze
            target_texts: List of texts to compare against
            
        Returns:
            Dictionary of analysis results
        """
        results = {
            "string_matches": [],
            "semantic_connections": [],
            "combined_score": 0.0
        }
        
        # Perform string matching analysis
        # Note: Assuming a proper implementation of StringMatchingModule
        # that would have similar methods to SemanticAnalysisModule
        
        # Perform semantic analysis
        semantic_connections = self.semantic_analyzer.find_semantic_connections(
            source_text, target_texts, threshold=0.5
        )
        
        results["semantic_connections"] = semantic_connections
        
        # Compute a combined relevance score
        if semantic_connections:
            results["combined_score"] = sum(c["similarity"] for c in semantic_connections) / len(semantic_connections)
        
        logger.info(f"Analysis complete. Found {len(semantic_connections)} semantic connections")
        return results
    
    def generate_report(self, analysis_results: Dict[str, Any], output_path: str = "report.txt") -> None:
        """
        Generate a text report from analysis results.
        
        Args:
            analysis_results: Results from analyze_texts
            output_path: Path to save the report
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as file:
                file.write("PALIMPSEST ANALYSIS REPORT\n")
                file.write("========================\n\n")
                
                file.write(f"Combined relevance score: {analysis_results['combined_score']:.4f}\n\n")
                
                file.write("SEMANTIC CONNECTIONS\n")
                file.write("-------------------\n")
                for i, conn in enumerate(analysis_results['semantic_connections']):
                    file.write(f"Connection {i+1}:\n")
                    file.write(f"  Similarity: {conn['similarity']:.4f}\n")
                    file.write(f"  Preview: {conn['text_preview']}\n\n")
            
            logger.info(f"Report generated and saved to {output_path}")
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise

def main():
    """Main function to run the Palimpsest application."""
    parser = argparse.ArgumentParser(description="Palimpsest Text Analysis Tool")
    parser.add_argument("source", help="Path to source text file")
    parser.add_argument("targets", nargs='+', help="Paths to target text files")
    parser.add_argument("--output", "-o", default="palimpsest_report.txt", 
                        help="Output report file path")
    
    args = parser.parse_args()
    
    try:
        analyzer = PalimpsestAnalyzer()
        
        # Load texts
        source_text = analyzer.load_text(args.source)
        target_texts = [analyzer.load_text(target) for target in args.targets]
        
        # Perform analysis
        results = analyzer.analyze_texts(source_text, target_texts)
        
        # Generate report
        analyzer.generate_report(results, args.output)
        
        print(f"Analysis complete. Report saved to {args.output}")
        return 0
    
    except Exception as e:
        logger.error(f"Error in Palimpsest analysis: {e}")
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())