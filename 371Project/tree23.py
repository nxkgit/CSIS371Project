import re

class Node23:
    """Node in a 2-3 tree"""
    def __init__(self):
        self.keys = []  # 1 or 2 keys (terms)
        self.posting_lists = []  # Corresponding posting lists
        self.children = []  # 0, 2, or 3 children
    
    def is_leaf(self):
        return len(self.children) == 0
    
    def is_full(self):
        return len(self.keys) == 2
    
    def insert_in_node(self, key, posting_list):
        """Insert key in sorted order within node"""
        i = 0
        while i < len(self.keys) and key > self.keys[i]:
            i += 1
        self.keys.insert(i, key)
        self.posting_lists.insert(i, posting_list)

class Tree23InvertedIndex:
    def __init__(self):
        self.root = Node23()
        self.stop_words = {'a', 'an', 'the', 'in', 'is', 'it', 'that', 'they', 
                           'can', 'be', 'will', 'but', 'such', 'also', 'have',
                           'if', 'at', 'to', 'as'}
    
    def tokenize(self, text):
        text = text.lower()
        tokens = re.findall(r'\b[a-z]+\b', text)
        return tokens
    
    def normalize(self, tokens):
        return [token for token in tokens if token not in self.stop_words]
    
    def search(self, term):
        """Search for a term and return its posting list"""
        return self._search_recursive(self.root, term)
    
    def _search_recursive(self, node, term):
        """Recursive search in 2-3 tree"""
        if node is None:
            return []
        
        # Check keys in current node
        for i, key in enumerate(node.keys):
            if term == key:
                return node.posting_lists[i]
        
        # If leaf and not found
        if node.is_leaf():
            return []
        
        # Find appropriate child to search
        if term < node.keys[0]:
            return self._search_recursive(node.children[0], term)
        elif len(node.keys) == 1 or term < node.keys[1]:
            return self._search_recursive(node.children[1], term)
        else:
            return self._search_recursive(node.children[2], term)
    
    def insert(self, term, doc_id):
        """Insert term into 2-3 tree"""
        # Check if term already exists
        existing = self.search(term)
        if existing:
            # Term exists, update posting list
            self._update_posting_list(self.root, term, doc_id)
        else:
            # New term, insert it
            result = self._insert_recursive(self.root, term, [doc_id])
            if result is not None:  # Root split
                new_root = Node23()
                new_root.keys = [result[1]]
                new_root.posting_lists = [[]]
                new_root.children = [result[0], result[2]]
                self.root = new_root
    
    def _update_posting_list(self, node, term, doc_id):
        """Update posting list for existing term"""
        for i, key in enumerate(node.keys):
            if term == key:
                if doc_id not in node.posting_lists[i]:
                    node.posting_lists[i].append(doc_id)
                return True
        
        if not node.is_leaf():
            if term < node.keys[0]:
                return self._update_posting_list(node.children[0], term, doc_id)
            elif len(node.keys) == 1 or term < node.keys[1]:
                return self._update_posting_list(node.children[1], term, doc_id)
            else:
                return self._update_posting_list(node.children[2], term, doc_id)
    
    def _insert_recursive(self, node, term, posting_list):
        """Recursive insertion with node splitting"""
        if node.is_leaf():
            if not node.is_full():
                node.insert_in_node(term, posting_list)
                return None
            else:
                # Split leaf node
                return self._split_node(node, term, posting_list, None)
        
        # Find child to insert into
        if term < node.keys[0]:
            child_idx = 0
        elif len(node.keys) == 1 or term < node.keys[1]:
            child_idx = 1
        else:
            child_idx = 2
        
        result = self._insert_recursive(node.children[child_idx], term, posting_list)
        
        if result is None:
            return None
        
        # Child split, need to insert middle value
        if not node.is_full():
            node.insert_in_node(result[1], [])
            node.children[child_idx] = result[0]
            node.children.insert(child_idx + 1, result[2])
            return None
        else:
            return self._split_node(node, result[1], [], result)
    
    def _split_node(self, node, new_key, new_posting, split_result):
        """Split a full node"""
        # Temporarily add new key
        temp_keys = node.keys + [new_key]
        temp_postings = node.posting_lists + [new_posting]
        temp_children = node.children[:]
        
        if split_result:
            # Insert split children
            for i, key in enumerate(temp_keys[:-1]):
                if new_key < key or (i == len(temp_keys) - 2):
                    idx = i if new_key < key else i + 1
                    temp_children[idx] = split_result[0]
                    temp_children.insert(idx + 1, split_result[2])
                    break
        
        # Sort
        sorted_pairs = sorted(zip(temp_keys, temp_postings))
        temp_keys = [k for k, _ in sorted_pairs]
        temp_postings = [p for _, p in sorted_pairs]
        
        # Create two new nodes
        left = Node23()
        left.keys = [temp_keys[0]]
        left.posting_lists = [temp_postings[0]]
        
        right = Node23()
        right.keys = [temp_keys[2]]
        right.posting_lists = [temp_postings[2]]
        
        if temp_children:
            left.children = temp_children[:2]
            right.children = temp_children[2:]
        
        return (left, temp_keys[1], right)
    
    def wildcard_search(self, pattern):
        """Search with wildcard"""
        if pattern.endswith('*'):
            prefix = pattern[:-1]
            results = []
            self._collect_with_prefix(self.root, prefix, results)
            return results
        return []
    
    def _collect_with_prefix(self, node, prefix, results):
        """Collect all terms starting with prefix"""
        if node is None:
            return
        
        for i, key in enumerate(node.keys):
            if key.startswith(prefix):
                results.append((key, node.posting_lists[i]))
        
        if not node.is_leaf():
            for child in node.children:
                self._collect_with_prefix(child, prefix, results)
    
    def add_document(self, doc_id, text):
        """Add document to index"""
        tokens = self.tokenize(text)
        terms = self.normalize(tokens)
        
        for term in terms:
            self.insert(term, doc_id)
    
    def display_index(self):
        """Display the index"""
        print("\n=== INVERTED INDEX (2-3 Tree) ===")
        terms = []
        self._collect_all_terms(self.root, terms)
        terms.sort()
        for term, posting_list in terms:
            docs = ', '.join(posting_list)
            print(f"{term} → [{docs}]")
    
    def _collect_all_terms(self, node, terms):
        """Collect all terms from tree"""
        if node is None:
            return
        
        for i, key in enumerate(node.keys):
            terms.append((key, node.posting_lists[i]))
        
        if not node.is_leaf():
            for child in node.children:
                self._collect_all_terms(child, terms)


# Main execution
if __name__ == "__main__":
    doc1 = """At very low temperatures, superconductors have zero resistance, 
              but they can also repel an external magnetic field, in such a way 
              that a spinning magnet can be held in a levitated position."""
    
    doc2 = """If a small magnet is brought near a superconductor, 
              it will be repelled."""
    
    index = Tree23InvertedIndex()
    index.add_document("Doc1", doc1)
    index.add_document("Doc2", doc2)
    
    index.display_index()
    
    print("\n=== SEARCH: s* ===")
    results = index.wildcard_search("s*")
    print(f"Terms found: {len(results)}")
    for term, docs in results:
        print(f"  {term}: {docs}")