# palimpsest/src/core/string_matching_module.py
"""
String Matching Module for Palimpsest.

This module provides capabilities for string matching and syntactic analysis:
- Suffix array implementation for efficient string pattern matching
- Locality-Sensitive Hashing (LSH) for approximate string matching
- Levenshtein distance calculation for edit distance
- Longest Common Substring detection
- N-gram analysis for text comparison

It works at the syntactic level to complement the semantic analysis module.
"""

import numpy as np
import re
from typing import List, Dict, Tuple, Union, Optional, Set, Any
from collections import defaultdict
import logging
import heapq
from difflib import SequenceMatcher

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StringMatchingModule:
    """Main class for string pattern matching and syntactic analysis."""
    
    def __init__(self, 
                 min_match_length: int = 5,
                 n_gram_size: int = 3,
                 lsh_bands: int = 20,
                 lsh_rows: int = 5,
                 case_sensitive: bool = False):
        """
        Initialize the string matching module.
        
        Args:
            min_match_length: Minimum length for string matches
            n_gram_size: Size of n-grams for analysis
            lsh_bands: Number of bands for LSH
            lsh_rows: Number of rows per band for LSH
            case_sensitive: Whether matches should be case sensitive
        """
        self.min_match_length = min_match_length
        self.n_gram_size = n_gram_size
        self.lsh_bands = lsh_bands
        self.lsh_rows = lsh_rows
        self.case_sensitive = case_sensitive
        logger.info(f"Initialized StringMatchingModule with min_match_length={min_match_length}")
        
        # Storage for suffix arrays
        self.suffix_arrays = {}
        
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for string matching.
        
        Args:
            text: Raw input text
            
        Returns:
            Preprocessed text
        """
        if not self.case_sensitive:
            text = text.lower()
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def build_suffix_array(self, text: str, text_id: str) -> List[int]:
        """
        Build a suffix array for efficient string matching.
        
        Args:
            text: Input text
            text_id: Identifier for this text
            
        Returns:
            Suffix array (list of indices)
        """
        n = len(text)
        
        # Generate all suffixes and their starting positions
        suffixes = [(text[i:], i) for i in range(n)]
        
        # Sort suffixes lexicographically
        suffixes.sort()
        
        # Extract the indices of the sorted suffixes
        suffix_array = [pos for _, pos in suffixes]
        
        # Store for later use
        self.suffix_arrays[text_id] = {
            'text': text,
            'suffix_array': suffix_array
        }
        
        logger.debug(f"Built suffix array for text {text_id}, length {n}")
        return suffix_array
    
    def find_exact_matches(self, pattern: str, text_id: str, 
                          min_length: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Find all exact occurrences of a pattern in text using suffix arrays.
        
        Args:
            pattern: Pattern to search for
            text_id: ID of text to search in
            min_length: Minimum match length (overrides default)
            
        Returns:
            List of match information dictionaries
        """
        if text_id not in self.suffix_arrays:
            logger.error(f"No suffix array found for text_id: {text_id}")
            return []
        
        suffix_data = self.suffix_arrays[text_id]
        text = suffix_data['text']
        suffix_array = suffix_data['suffix_array']
        
        min_match = min_length if min_length is not None else self.min_match_length
        
        if len(pattern) < min_match:
            logger.warning(f"Pattern length {len(pattern)} is less than min_match {min_match}")
            return []
        
        # Binary search to find pattern in suffix array
        left, right = 0, len(suffix_array) - 1
        matches = []
        
        # Process pattern
        pattern = self.preprocess_text(pattern) if not self.case_sensitive else pattern
        
        # Find first occurrence
        first = -1
        while left <= right:
            mid = (left + right) // 2
            suffix_pos = suffix_array[mid]
            suffix = text[suffix_pos:]
            
            if suffix.startswith(pattern):
                first = mid
                right = mid - 1
            elif suffix < pattern:
                left = mid + 1
            else:
                right = mid - 1
        
        # If pattern not found
        if first == -1:
            return []
        
        # Find last occurrence
        left, right = first, len(suffix_array) - 1
        last = first
        while left <= right:
            mid = (left + right) // 2
            suffix_pos = suffix_array[mid]
            suffix = text[suffix_pos:]
            
            if suffix.startswith(pattern):
                last = mid
                left = mid + 1
            else:
                right = mid - 1
        
        # Collect all matches
        for i in range(first, last + 1):
            pos = suffix_array[i]
            matches.append({
                'position': pos,
                'end_position': pos + len(pattern),
                'match': text[pos:pos+len(pattern)],
                'context_before': text[max(0, pos-30):pos],
                'context_after': text[pos+len(pattern):min(len(text), pos+len(pattern)+30)]
            })
        
        logger.info(f"Found {len(matches)} exact matches for pattern in text {text_id}")
        return matches
    
    def compute_levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Compute Levenshtein (edit) distance between two strings.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Edit distance (lower means more similar)
        """
        # Create distance matrix
        m, n = len(s1), len(s2)
        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
        
        # Initialize first row and column
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
            
        # Fill dp matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j],      # deletion
                                      dp[i][j-1],      # insertion
                                      dp[i-1][j-1])    # substitution
                                      
        return dp[m][n]
    
    def find_longest_common_substring(self, s1: str, s2: str) -> Tuple[str, int, int]:
        """
        Find the longest common substring between two strings.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Tuple of (substring, start_pos_s1, start_pos_s2)
        """
        # Preprocess
        if not self.case_sensitive:
            s1 = s1.lower()
            s2 = s2.lower()
            
        # Use SequenceMatcher for efficient LCS finding
        matcher = SequenceMatcher(None, s1, s2)
        match = matcher.find_longest_match(0, len(s1), 0, len(s2))
        
        substring = s1[match.a:match.a + match.size]
        return substring, match.a, match.b
    
    def generate_n_grams(self, text: str, n: Optional[int] = None) -> List[str]:
        """
        Generate n-grams from text.
        
        Args:
            text: Input text
            n: Size of n-grams (uses default if None)
            
        Returns:
            List of n-grams
        """
        if n is None:
            n = self.n_gram_size
            
        # Preprocess
        text = self.preprocess_text(text)
        
        # Generate word n-grams
        words = text.split()
        n_grams = []
        
        for i in range(len(words) - n + 1):
            n_gram = ' '.join(words[i:i+n])
            n_grams.append(n_gram)
            
        return n_grams
    
    def compute_jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """
        Compute Jaccard similarity between two sets.
        
        Args:
            set1: First set
            set2: Second set
            
        Returns:
            Jaccard similarity (0-1 range)
        """
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0.0
            
        return intersection / union
    
    def _hash_n_gram(self, n_gram: str, num_hashes: int = 100) -> List[int]:
        """
        Create hash values for an n-gram for LSH.
        
        Args:
            n_gram: Input n-gram
            num_hashes: Number of hash functions
            
        Returns:
            List of hash values
        """
        hashes = []
        
        # Simple hashing scheme using different prime multipliers
        for i in range(num_hashes):
            prime = 31 + i * 2
            hash_val = 0
            
            for char in n_gram:
                hash_val = (hash_val * prime + ord(char)) & 0xFFFFFFFF
                
            hashes.append(hash_val)
            
        return hashes
    
    def create_lsh_signature(self, n_grams: List[str]) -> List[int]:
        """
        Create LSH signature matrix for a set of n-grams.
        
        Args:
            n_grams: List of n-grams
            
        Returns:
            LSH signature (list of min-hash values)
        """
        n_gram_set = set(n_grams)
        total_hashes = self.lsh_bands * self.lsh_rows
        
        # Initialize signature with max values
        signature = [float('inf')] * total_hashes
        
        # For each n-gram, compute hash values and keep minimum
        for n_gram in n_gram_set:
            hashes = self._hash_n_gram(n_gram, total_hashes)
            
            for i, hash_val in enumerate(hashes):
                signature[i] = min(signature[i], hash_val)
                
        return signature
    
    def lsh_compare_texts(self, text1: str, text2: str) -> float:
        """
        Compare two texts using Locality Sensitive Hashing.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1 range)
        """
        # Generate n-grams for each text
        ngrams1 = self.generate_n_grams(text1)
        ngrams2 = self.generate_n_grams(text2)
        
        # Create LSH signatures
        sig1 = self.create_lsh_signature(ngrams1)
        sig2 = self.create_lsh_signature(ngrams2)
        
        # Count matching bands
        matching_bands = 0
        
        for band in range(self.lsh_bands):
            start_idx = band * self.lsh_rows
            end_idx = start_idx + self.lsh_rows
            
            band1 = tuple(sig1[start_idx:end_idx])
            band2 = tuple(sig2[start_idx:end_idx])
            
            if band1 == band2:
                matching_bands += 1
                
        # Compute similarity based on matching bands
        similarity = matching_bands / self.lsh_bands
        
        return similarity
    
    def find_text_overlaps(self, text1: str, text2: str, 
                         min_overlap_length: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Find all substantial text overlaps between two texts.
        
        Args:
            text1: First text
            text2: Second text
            min_overlap_length: Minimum length for overlaps
            
        Returns:
            List of overlap information dictionaries
        """
        min_length = min_overlap_length if min_overlap_length else self.min_match_length
        
        # Preprocess texts
        text1 = self.preprocess_text(text1)
        text2 = self.preprocess_text(text2)
        
        # Build suffix array for text2
        text_id = "overlap_search"
        self.build_suffix_array(text2, text_id)
        
        overlaps = []
        
        # Search for chunks of text1 in text2
        chunk_size = max(100, min_length * 2)
        
        for i in range(0, len(text1), chunk_size // 2):  # 50% overlap between chunks
            chunk = text1[i:i+chunk_size]
            
            if len(chunk) < min_length:
                continue
                
            # Find matches for this chunk
            matches = self.find_exact_matches(chunk, text_id, min_length)
            
            for match in matches:
                # Extend match as far as possible
                t1_start = i
                t2_start = match['position']
                
                # Find how far the match extends backward
                backward = 0
                while (t1_start - backward > 0 and 
                       t2_start - backward > 0 and 
                       text1[t1_start - backward - 1] == text2[t2_start - backward - 1]):
                    backward += 1
                
                # Find how far the match extends forward
                forward = len(match['match'])
                while (t1_start + forward < len(text1) and 
                       t2_start + forward < len(text2) and
                       text1[t1_start + forward] == text2[t2_start + forward]):
                    forward += 1
                
                # Calculate extended match boundaries
                t1_extended_start = t1_start - backward
                t2_extended_start = t2_start - backward
                extended_length = backward + forward
                
                # Add if it meets minimum length
                if extended_length >= min_length:
                    match_text = text1[t1_extended_start:t1_extended_start + extended_length]
                    
                    overlaps.append({
                        'text1_start': t1_extended_start,
                        'text1_end': t1_extended_start + extended_length,
                        'text2_start': t2_extended_start,
                        'text2_end': t2_extended_start + extended_length,
                        'length': extended_length,
                        'overlap_text': match_text
                    })
        
        # Remove duplicates and sort by length
        unique_overlaps = []
        seen = set()
        
        for overlap in sorted(overlaps, key=lambda x: -x['length']):
            key = (overlap['text1_start'], overlap['text2_start'], overlap['length'])
            if key not in seen:
                seen.add(key)
                unique_overlaps.append(overlap)
                
        logger.info(f"Found {len(unique_overlaps)} unique overlaps between texts")
        return unique_overlaps
    
    def compute_string_similarity(self, text1: str, text2: str) -> Dict[str, float]:
        """
        Compute string similarity metrics between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Dictionary of similarity metrics
        """
        # Preprocess
        t1 = self.preprocess_text(text1)
        t2 = self.preprocess_text(text2)
        
        # Calculate similarity metrics
        results = {}
        
        # 1. N-gram Jaccard similarity
        ngrams1 = set(self.generate_n_grams(t1))
        ngrams2 = set(self.generate_n_grams(t2))
        results['ngram_jaccard'] = self.compute_jaccard_similarity(ngrams1, ngrams2)
        
        # 2. Longest common substring proportion
        lcs, _, _ = self.find_longest_common_substring(t1, t2)
        max_length = max(len(t1), len(t2))
        results['lcs_proportion'] = len(lcs) / max_length if max_length > 0 else 0
        
        # 3. Normalized Levenshtein distance
        if max(len(t1), len(t2)) <= 1000:  # Limit for performance reasons
            edit_distance = self.compute_levenshtein_distance(t1[:1000], t2[:1000])
            results['levenshtein_similarity'] = 1 - (edit_distance / max(len(t1[:1000]), len(t2[:1000])))
        else:
            # Use LSH for longer texts
            results['levenshtein_similarity'] = None
            
        # 4. LSH similarity
        results['lsh_similarity'] = self.lsh_compare_texts(t1, t2)
        
        # 5. SequenceMatcher ratio (difflib)
        results['sequence_matcher'] = SequenceMatcher(None, t1[:5000], t2[:5000]).ratio()
        
        # 6. Overall similarity score (weighted average)
        weights = {
            'ngram_jaccard': 0.25,
            'lcs_proportion': 0.2,
            'sequence_matcher': 0.3,
            'lsh_similarity': 0.25
        }
        
        overall_score = 0
        weight_sum = 0
        
        for metric, weight in weights.items():
            if results[metric] is not None:
                overall_score += results[metric] * weight
                weight_sum += weight
                
        if weight_sum > 0:
            results['overall_score'] = overall_score / weight_sum
        else:
            results['overall_score'] = 0
            
        logger.info(f"String similarity overall score: {results['overall_score']:.4f}")
        return results
    
    def find_repeated_phrases(self, text: str, min_length: int = 10, 
                             min_repetitions: int = 2) -> List[Dict[str, Any]]:
        """
        Find phrases that are repeated within a single text.
        
        Args:
            text: Text to analyze
            min_length: Minimum phrase length to consider
            min_repetitions: Minimum number of repetitions
            
        Returns:
            List of dictionaries with repeated phrase information
        """
        # Preprocess text
        processed = self.preprocess_text(text)
        
        # Build suffix array
        text_id = "repeated_phrases"
        suffix_array = self.build_suffix_array(processed, text_id)
        
        # Find repeated phrases by looking for common prefixes between adjacent suffixes
        phrases = defaultdict(list)
        
        for i in range(len(suffix_array) - 1):
            pos1 = suffix_array[i]
            pos2 = suffix_array[i+1]
            
            # Calculate length of common prefix
            j = 0
            while (pos1 + j < len(processed) and 
                   pos2 + j < len(processed) and 
                   processed[pos1 + j] == processed[pos2 + j]):
                j += 1
                
                # Record when we hit minimum length
                if j >= min_length:
                    phrase = processed[pos1:pos1+j]
                    phrases[phrase].append(pos1)
                    break
        
        # Filter for phrases with enough repetitions and build result
        results = []
        
        for phrase, positions in phrases.items():
            if len(positions) + 1 >= min_repetitions:  # +1 because we only recorded first position
                # Find all occurrences
                all_positions = []
                
                # Use suffix array to find all occurrences efficiently
                for match in self.find_exact_matches(phrase, text_id):
                    all_positions.append(match['position'])
                
                results.append({
                    'phrase': phrase,
                    'length': len(phrase),
                    'positions': sorted(all_positions),
                    'repetitions': len(all_positions)
                })
                
        # Sort by number of repetitions and then length
        results.sort(key=lambda x: (-x['repetitions'], -x['length']))
        
        logger.info(f"Found {len(results)} repeated phrases with at least {min_repetitions} repetitions")
        return results