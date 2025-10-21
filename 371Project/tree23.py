import re
from typing import Dict, Set, List


class boolean_model:
    """Boolean model with inverted index and permuterm support for single-* wildcards.

    Public API:
      - __init__(documents: Dict[str, str])
      - boolean_query(query: str) -> List[str]
    """

    def __init__(self, documents: Dict[str, str]):
        self.documents = documents
        self.stop_words = {
            'a', 'an', 'the', 'in', 'is', 'it', 'that', 'they', 'can', 'be', 'will',
            'but', 'such', 'also', 'have', 'if', 'at', 'to', 'as', 'near', 'very',
        }
        self.inverted: Dict[str, Set[str]] = {}  # term -> set(doc_ids)
        self.permuterm: Dict[str, Set[str]] = {}  # rotation -> set(terms)
        self._build_indexes()

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        return re.findall(r"\b[a-z]+\b", text)

    def _normalize(self, tokens: List[str]) -> List[str]:
        return [t for t in tokens if t not in self.stop_words]

    def _add_term(self, term: str, doc_id: str):
        self.inverted.setdefault(term, set()).add(doc_id)

    def _add_permuterms_for(self, term: str):
        t = term + '$'
        for i in range(len(t)):
            rot = t[i:] + t[:i]
            self.permuterm.setdefault(rot, set()).add(term)

    def _build_indexes(self):
        for doc_id, text in self.documents.items():
            tokens = self._tokenize(text)
            terms = self._normalize(tokens)
            for term in terms:
                self._add_term(term, doc_id)

        # Build permuterm from all terms
        for term in list(self.inverted.keys()):
            self._add_permuterms_for(term)

    def _wildcard_match(self, pattern: str) -> Set[str]:
        """Match single-* wildcard patterns using the permuterm index.

        Only patterns with zero or one '*' are supported. For example:
          - 'super*' or 's*' or '*man' or 'su*er' (single star anywhere)
        """
        if '*' not in pattern:
            return {pattern} if pattern in self.inverted else set()

        if pattern.count('*') > 1:
            return set()

        pre, post = pattern.split('*')
        # rotated search string is post + '$' + pre
        rotated = post + '$' + pre
        matches = set()
        for rot_key, terms in self.permuterm.items():
            if rot_key.startswith(rotated):
                matches.update(terms)
        return matches

    def _get_postings_for_term_or_pattern(self, token: str) -> Set[str]:
        if '*' in token:
            terms = self._wildcard_match(token)
            postings = set()
            for t in terms:
                postings.update(self.inverted.get(t, set()))
            return postings
        return set(self.inverted.get(token, set()))

    def boolean_query(self, query: str) -> List[str]:
        """Process a boolean query with up to two terms and one operator.

        Supported operators: AND, OR, NOT (binary as A NOT B -> A AND NOT B), XOR,
        AND NOT, OR NOT, and unary NOT ("NOT term").

        Returns a sorted list of matching document IDs.
        """
        q = query.strip()
        if not q:
            return []

        tokens = q.split()

        # recognize two-word operators first (AND NOT, OR NOT)
        op = None
        term1 = None
        term2 = None

        if len(tokens) >= 3 and tokens[1].upper() == 'AND' and tokens[2].upper() == 'NOT':
            term1 = tokens[0].lower()
            op = 'AND NOT'
            term2 = tokens[3].lower() if len(tokens) > 3 else ''
        elif len(tokens) >= 3 and tokens[1].upper() == 'OR' and tokens[2].upper() == 'NOT':
            term1 = tokens[0].lower()
            op = 'OR NOT'
            term2 = tokens[3].lower() if len(tokens) > 3 else ''
        elif len(tokens) == 1:
            term1 = tokens[0].lower()
            op = None
        elif len(tokens) == 2 and tokens[0].upper() == 'NOT':
            op = 'UNARY NOT'
            term1 = tokens[1].lower()
        elif len(tokens) >= 3:
            term1 = tokens[0].lower()
            op = tokens[1].upper()
            term2 = tokens[2].lower()
        else:
            return []

        universe = set(self.documents.keys())

        if op is None:
            res = self._get_postings_for_term_or_pattern(term1)
            return sorted(res)

        if op == 'UNARY NOT':
            postings = self._get_postings_for_term_or_pattern(term1)
            return sorted(universe - postings)

        postings1 = self._get_postings_for_term_or_pattern(term1) if term1 else set()
        postings2 = self._get_postings_for_term_or_pattern(term2) if term2 else set()

        if op == 'AND':
            result = postings1 & postings2
        elif op == 'OR':
            result = postings1 | postings2
        elif op == 'NOT':
            result = postings1 - postings2
        elif op == 'XOR':
            result = postings1 ^ postings2
        elif op == 'AND NOT':
            result = postings1 - postings2
        elif op == 'OR NOT':
            result = postings1 | (universe - postings2)
        else:
            result = set()

        return sorted(result)


if __name__ == '__main__':
    # Example documents
    docs = {
        'Doc1': (
            "At very low temperatures, superconductors have zero resistance, "
            "but they can also repel an external magnetic field, in such a way "
            "that a spinning magnet can be held in a levitated position."
        ),
        'Doc2': (
            "If a small magnet is brought near a superconductor, it will be repelled."
        ),
    }

    bm = boolean_model(docs)

    examples = [
        'superconductor AND magnet',
        'magnet AND NOT superconductor',
        'superconductor OR magnet',
        'superconductor XOR magnet',
        's*',
        'super* AND magnet',
        'NOT magnet',
        'magnet OR NOT superconductor',
    ]

    print('\n=== Inverted index terms and postings ===')
    for term in sorted(bm.inverted.keys()):
        print(f"{term} -> {sorted(bm.inverted[term])}")

    print('\n=== Example queries ===')
    for q in examples:
        res = bm.boolean_query(q)
        print(f"Query: {q} -> {res}")

