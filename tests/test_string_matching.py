# palimpsest/tests/test_string_matching.py
import unittest
import sys
import os
from typing import List, Dict, Any

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.string_matching_module import StringMatchingModule

class TestStringMatchingModule(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.matcher = StringMatchingModule(
            min_match_length=4,
            n_gram_size=3,
            lsh_bands=10,
            lsh_rows=4,
            case_sensitive=False
        )
        
        # Sample texts for testing
        self.text1 = "To be or not to be, that is the question. Whether 'tis nobler in the mind to suffer."
        self.text2 = "That is the question indeed! To be or not to be. Whether suffering is noble is debatable."
        self.text3 = "Completely different text with no overlap whatsoever to the others."

    def test_preprocess_text(self):
        """Test text preprocessing."""
        text = "  Multiple   spaces and\tTABS\nand newlines  "
        expected = "multiple spaces and tabs and newlines"
        self.assertEqual(self.matcher.preprocess_text(text), expected)

    def test_build_suffix_array(self):
        """Test suffix array creation."""
        text = "banana"
        text_id = "test"
        suffix_array = self.matcher.build_suffix_array(text, text_id)
        
        # Expected suffix array for "banana" is [5, 3, 1, 0, 4, 2]
        # Suffixes in lexicographical order: a, ana, anana, banana, na, nana
        expected_array = [5, 3, 1, 0, 4, 2]
        self.assertEqual(suffix_array, expected_array)
        
        # Check that the suffix array was stored correctly
        self.assertIn(text_id, self.matcher.suffix_arrays)
        self.assertEqual(self.matcher.suffix_arrays[text_id]['text'], text)

    def test_find_exact_matches(self):
        """Test finding exact matches using suffix arrays."""
        text = "banana banana banana"
        text_id = "test_exact"
        self.matcher.build_suffix_array(text, text_id)
        
        # Find "ana" in "banana banana banana"
        matches = self.matcher.find_exact_matches("ana", text_id)
        
        # Should find "ana" at positions 3, 10, 17
        self.assertEqual(len(matches), 3)
        positions = [match['position'] for match in matches]
        self.assertIn(3, positions)
        self.assertIn(10, positions)
        self.assertIn(17, positions)
        
        # Try non-existent pattern
        matches = self.matcher.find_exact_matches("xyz", text_id)
        self.assertEqual(len(matches), 0)

    def test_compute_levenshtein_distance(self):
        """Test Levenshtein distance calculation."""
        # Test cases
        test_cases = [
            ("kitten", "sitting", 3),  # 3 operations: k→s, e→i, ∅→g
            ("saturday", "sunday", 3),  # 3 operations: remove 'a', remove 't', a→u
            ("", "hello", 5),          # 5 operations: insert all 5 chars
            ("hello", "hello", 0),     # 0 operations: already identical
        ]
        
        for s1, s2, expected in test_cases:
            self.assertEqual(self.matcher.compute_levenshtein_distance(s1, s2), expected)

    def test_find_longest_common_substring(self):
        """Test finding longest common substring."""
        test_cases = [
            ("abcdefg", "xyzabcde", "abcde"),
            ("programming", "gaming", "ing"),
            ("abcdef", "xyz", ""),
        ]
        
        for s1, s2, expected_lcs in test_cases:
            lcs, _, _ = self.matcher.find_longest_common_substring(s1, s2)
            self.assertEqual(lcs, expected_lcs)

    def test_generate_n_grams(self):
        """Test n-gram generation."""
        text = "this is a test sentence for n-grams"
        
        # Test default n-gram size (3)
        default_ngrams = self.matcher.generate_n_grams(text)
        expected_default = [
            "this is a", "is a test", "a test sentence", 
            "test sentence for", "sentence for n-grams"
        ]
        self.assertEqual(default_ngrams, expected_default)
        
        # Test with n=2
        bigrams = self.matcher.generate_n_grams(text, 2)
        expected_bigrams = ["this is", "is a", "a test", "test sentence", "sentence for", "for n-grams"]
        self.assertEqual(bigrams, expected_bigrams)

    def test_compute_jaccard_similarity(self):
        """Test Jaccard similarity calculation."""
        set1 = {"apple", "banana", "cherry"}
        set2 = {"banana", "cherry", "date"}
        
        # Intersection: banana, cherry (2 elements)
        # Union: apple, banana, cherry, date (4 elements)
        # Jaccard similarity = 2/4 = 0.5
        expected_similarity = 0.5
        
        similarity = self.matcher.compute_jaccard_similarity(set1, set2)
        self.assertEqual(similarity, expected_similarity)
        
        # Test edge cases
        self.assertEqual(self.matcher.compute_jaccard_similarity(set(), set()), 0.0)
        self.assertEqual(self.matcher.compute_jaccard_similarity({"a"}, {"a"}), 1.0)

    def test_lsh_compare_texts(self):
        """Test LSH text comparison."""
        # Similar texts should have higher similarity
        sim_1_2 = self.matcher.lsh_compare_texts(self.text1, self.text2)
        sim_1_3 = self.matcher.lsh_compare_texts(self.text1, self.text3)
        
        # Text1 and Text2 share phrases, should have higher similarity
        self.assertGreater(sim_1_2, sim_1_3)
        
        # Same text should have maximum similarity
        self.assertEqual(self.matcher.lsh_compare_texts(self.text1, self.text1), 1.0)

    def test_find_text_overlaps(self):
        """Test finding overlaps between texts."""
        overlaps = self.matcher.find_text_overlaps(self.text1, self.text2)
        
        # Should find "To be or not to be" and "that is the question" as overlaps
        self.assertGreater(len(overlaps), 0)
        
        # Check that each overlap meets minimum length
        for overlap in overlaps:
            self.assertGreaterEqual(overlap['length'], self.matcher.min_match_length)
            
        # Check no overlaps with text3
        overlaps_with_3 = self.matcher.find_text_overlaps(self.text1, self.text3)
        self.assertEqual(len(overlaps_with_3), 0)

    def test_compute_string_similarity(self):
        """Test computing overall string similarity metrics."""
        similarity = self.matcher.compute_string_similarity(self.text1, self.text2)
        
        # Check that all expected metrics are present
        expected_metrics = [
            'ngram_jaccard', 'lcs_proportion', 'sequence_matcher', 
            'lsh_similarity', 'overall_score'
        ]
        for metric in expected_metrics:
            self.assertIn(metric, similarity)
            self.assertIsNotNone(similarity[metric])
            
        # Similar texts should have positive scores
        self.assertGreater(similarity['overall_score'], 0)
        
        # Different texts should have lower scores
        similarity_different = self.matcher.compute_string_similarity(self.text1, self.text3)
        self.assertLess(similarity_different['overall_score'], similarity['overall_score'])

    def test_find_repeated_phrases(self):
        """Test finding repeated phrases within a text."""
        # Text with repeated phrases
        repetitive_text = "The quick brown fox jumps over the lazy dog. " * 3
        repeated = self.matcher.find_repeated_phrases(repetitive_text, min_length=5, min_repetitions=2)
        
        # Should find "The quick brown fox jumps over the lazy dog" repeated
        self.assertGreater(len(repeated), 0)
        
        # Check that repetitions match expectations
        for phrase_info in repeated:
            self.assertGreaterEqual(phrase_info['repetitions'], 2)
            self.assertGreaterEqual(phrase_info['length'], 5)

if __name__ == '__main__':
    unittest.main()