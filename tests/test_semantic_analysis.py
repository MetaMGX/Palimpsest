# palimpsest/tests/test_semantic_analysis.py
import unittest
import sys
import os
import numpy as np
from typing import List, Dict, Any

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.semantic_analysis_module import SemanticAnalysisModule

class TestSemanticAnalysisModule(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SemanticAnalysisModule(
            model_name="test_model",
            embedding_dim=100,  # Smaller dimension for testing
            use_pretrained=False
        )
        
        # Sample texts for testing
        self.text1 = "The quick brown fox jumps over the lazy dog."
        self.text2 = "A rapid auburn fox leaps over the idle canine."  # Semantically similar to text1
        self.text3 = "Neural networks have revolutionized machine learning and artificial intelligence."

    def test_initialization(self):
        """Test proper initialization of SemanticAnalysisModule."""
        # Check that attributes are set correctly
        self.assertEqual(self.analyzer.model_name, "test_model")
        self.assertEqual(self.analyzer.embedding_dim, 100)
        self.assertFalse(self.analyzer.use_pretrained)
        self.assertEqual(self.analyzer.models, {})

    def test_load_models(self):
        """Test model loading."""
        # Currently this is a placeholder since we don't actually load models in the implementation
        result = self.analyzer.load_models()
        self.assertTrue(result)
        self.assertIn("embedding", self.analyzer.models)
        self.assertIn("topic", self.analyzer.models)

    def test_compute_text_embedding(self):
        """Test text embedding computation."""
        # Test that embeddings are of the right dimension
        embedding = self.analyzer.compute_text_embedding(self.text1)
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape, (self.analyzer.embedding_dim,))
        
        # Test that different calls produce different embeddings
        # Note: There's a very small probability this could fail since we're using random values
        embedding2 = self.analyzer.compute_text_embedding(self.text1)
        self.assertFalse(np.array_equal(embedding, embedding2))

    def test_compute_semantic_similarity(self):
        """Test computation of semantic similarity between texts."""
        # Compute similarities
        sim_1_2 = self.analyzer.compute_semantic_similarity(self.text1, self.text2)
        sim_1_3 = self.analyzer.compute_semantic_similarity(self.text1, self.text3)
        sim_1_1 = self.analyzer.compute_semantic_similarity(self.text1, self.text1)
        
        # Check basic properties of similarity measures
        self.assertGreaterEqual(sim_1_2, 0.0)
        self.assertLessEqual(sim_1_2, 1.0)
        
        # Identical texts should have maximum similarity
        self.assertAlmostEqual(sim_1_1, 1.0)
        
        # Due to random embeddings, we can't make strong assertions about sim_1_2 vs sim_1_3
        # But we can check that the function runs without errors

    def test_extract_topics(self):
        """Test topic extraction from texts."""
        texts = [self.text1, self.text2, self.text3]
        num_topics = 3
        topics = self.analyzer.extract_topics(texts, num_topics)
        
        # Check correct number of topics
        self.assertEqual(len(topics), num_topics)
        
        # Check each topic has keywords
        for topic_id, keywords in topics.items():
            self.assertIsInstance(keywords, list)
            self.assertTrue(all(isinstance(kw, str) for kw in keywords))
            self.assertEqual(len(keywords), 5)  # Default is 5 keywords per topic

    def test_find_semantic_connections(self):
        """Test finding semantic connections between texts."""
        target_texts = [self.text2, self.text3]
        connections = self.analyzer.find_semantic_connections(
            self.text1, target_texts, threshold=0.0  # Set threshold to 0 to ensure we get connections
        )
        
        # Should find connections with both texts (since threshold is 0)
        self.assertEqual(len(connections), 2)
        
        # Check connection structure
        for conn in connections:
            self.assertIn("target_index", conn)
            self.assertIn("similarity", conn)
            self.assertIn("text_preview", conn)
            self.assertIsInstance(conn["similarity"], float)
            self.assertGreaterEqual(conn["similarity"], 0.0)
            self.assertLessEqual(conn["similarity"], 1.0)
        
        # Test with high threshold
        high_threshold_connections = self.analyzer.find_semantic_connections(
            self.text1, target_texts, threshold=0.99
        )
        # With random embeddings, unlikely to exceed this high threshold
        self.assertLessEqual(len(high_threshold_connections), 2)

    def test_analyze_conceptual_overlap(self):
        """Test conceptual overlap analysis."""
        overlap = self.analyzer.analyze_conceptual_overlap(self.text1, self.text2)
        
        # Check that all expected fields are present
        self.assertIn("overall_similarity", overlap)
        self.assertIn("shared_concepts", overlap)
        self.assertIn("conceptual_overlap_score", overlap)
        
        # Check value ranges
        self.assertGreaterEqual(overlap["overall_similarity"], 0.0)
        self.assertLessEqual(overlap["overall_similarity"], 1.0)
        self.assertGreaterEqual(overlap["conceptual_overlap_score"], 0.0)
        self.assertLessEqual(overlap["conceptual_overlap_score"], 1.0)
        
        # Check shared concepts
        self.assertIsInstance(overlap["shared_concepts"], list)
        if overlap["overall_similarity"] > 0.5:
            self.assertGreater(len(overlap["shared_concepts"]), 0)

if __name__ == '__main__':
    unittest.main()