import re
import math
from typing import Dict, List, Set, Tuple
from collections import Counter, defaultdict


class VectorModel:
    """Vector space model with tf-idf weighting and cosine normalization."""

    def __init__(self, documents: Dict[str, str]):
        self.documents = documents
        self.stop_words = {
            'a', 'an', 'the', 'in', 'is', 'it', 'that', 'they', 'can', 'be', 'will',
            'but', 'such', 'also', 'have', 'if', 'at', 'to', 'as', 'near', 'very',
            'for', 'while', 'and', 'or', 'are', 'by', 'of'
        }
        # Term -> {doc_id: (tf, tf_log, weight)}
        self.inverted: Dict[str, Dict[str, Tuple[int, float, float]]] = {}
        self.doc_lengths: Dict[str, float] = {}  # Document lengths for normalization
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize and normalize text."""
        text = text.lower()
        return re.findall(r"\b[a-z]+\b", text)

    def _normalize(self, tokens: List[str]) -> List[str]:
        """Remove stop words."""
        return [t for t in tokens if t not in self.stop_words]

    def _compute_tf_log(self, tf: int) -> float:
        """Compute logarithmic term frequency."""
        return 1 + math.log10(tf) if tf > 0 else 0

    def _compute_idf(self, term: str) -> float:
        """Compute inverse document frequency."""
        df = len(self.inverted[term])
        return math.log10(len(self.documents) / df) if df > 0 else 0

    def _build_index(self):
        """Build inverted index with tf, tf-log, and weights."""
        # First pass: compute term frequencies
        doc_term_freqs = defaultdict(lambda: defaultdict(int))
        for doc_id, text in self.documents.items():
            tokens = self._tokenize(text)
            terms = self._normalize(tokens)
            for term in terms:
                doc_term_freqs[term][doc_id] += 1

        # Second pass: compute tf-log and prepare for weights
        for term, doc_freqs in doc_term_freqs.items():
            self.inverted[term] = {}
            for doc_id, tf in doc_freqs.items():
                tf_log = self._compute_tf_log(tf)
                self.inverted[term][doc_id] = (tf, tf_log, 0)  # weight will be updated later

        # Third pass: compute document lengths and normalize weights
        doc_vectors = defaultdict(dict)
        for term in self.inverted:
            for doc_id, (_, tf_log, _) in self.inverted[term].items():
                doc_vectors[doc_id][term] = tf_log

        # Compute document lengths for normalization
        for doc_id, term_weights in doc_vectors.items():
            length = math.sqrt(sum(w * w for w in term_weights.values()))
            self.doc_lengths[doc_id] = length

            # Update normalized weights in inverted index
            for term, tf_log in term_weights.items():
                tf, _, _ = self.inverted[term][doc_id]
                normalized_weight = tf_log / length if length > 0 else 0
                self.inverted[term][doc_id] = (tf, tf_log, normalized_weight)

    def process_query(self, query: str) -> List[Tuple[str, float]]:
        """Process a query and return ranked documents with scores."""
        # Tokenize and normalize query
        tokens = self._tokenize(query)
        query_terms = self._normalize(tokens)
        
        # Count term frequencies in query
        query_tf = Counter(query_terms)
        
        # Compute query weights (tf-idf)
        query_weights = {}
        query_length = 0
        for term in set(query_terms):
            if term in self.inverted:
                tf = query_tf[term]
                tf_log = self._compute_tf_log(tf)
                idf = self._compute_idf(term)
                weight = tf_log * idf
                query_weights[term] = weight
                query_length += weight * weight
        
        # Normalize query weights
        query_length = math.sqrt(query_length)
        if query_length > 0:
            for term in query_weights:
                query_weights[term] /= query_length

        # Calculate document scores
        doc_scores = defaultdict(float)
        for term, query_weight in query_weights.items():
            for doc_id, (_, _, doc_weight) in self.inverted[term].items():
                doc_scores[doc_id] += query_weight * doc_weight

        # Sort documents by score
        ranked_docs = sorted(doc_scores.items(), key=lambda x: (-x[1], x[0]))
        return ranked_docs

    def output_trec_format(self, query_id: str, ranked_docs: List[Tuple[str, float]], 
                         run_id: str = "VectorModel") -> List[str]:
        """Convert ranked results to TREC format."""
        trec_lines = []
        for rank, (doc_id, score) in enumerate(ranked_docs, 1):
            # Format: query_id Q0 doc_id rank score run_id
            trec_line = f"{query_id} Q0 {doc_id} {rank} {score:.6f} {run_id}"
            trec_lines.append(trec_line)
        return trec_lines


if __name__ == '__main__':
    # Example documents from the assignment
    docs = {
        'd1': "for english model retireval have a relevance model while vector space model retrieval",
        'd2': "r-precision measure is relevant to average precision measure",
        'd3': "most efficient retrieval models are language model and vector space model",
        'd4': "english is the most efficient language",
        'd5': "retrieval efficiency is measured by average precision"
    }

    # Create vector model
    vm = VectorModel(docs)

    # Process query
    query = "effici* retrieval model"
    results = vm.process_query(query)

    # Output in TREC format
    trec_output = vm.output_trec_format("Q1", results)
    
    print("\nQuery:", query)
    print("\nRanked results in TREC format:")
    for line in trec_output:
        print(line)