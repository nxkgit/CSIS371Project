import unittest
from tree23 import boolean_model

DOCS = {
    'Doc1': (
        "At very low temperatures, superconductors have zero resistance, "
        "but they can also repel an external magnetic field, in such a way "
        "that a spinning magnet can be held in a levitated position."
    ),
    'Doc2': (
        "If a small magnet is brought near a superconductor, it will be repelled."
    ),
}

class TestBooleanModel(unittest.TestCase):
    def setUp(self):
        self.bm = boolean_model(DOCS)

    def assertQuery(self, query, expected_set):
        res = set(self.bm.boolean_query(query))
        self.assertEqual(res, set(expected_set), msg=f"Query: {query} -> {res}, expected {expected_set}")

    def test_and(self):
        self.assertQuery('superconductor AND magnet', ['Doc2'])

    def test_and_not(self):
        self.assertQuery('magnet AND NOT superconductor', ['Doc1'])

    def test_or(self):
        self.assertQuery('superconductor OR magnet', ['Doc1','Doc2'])

    def test_xor(self):
        self.assertQuery('superconductor XOR magnet', ['Doc1'])

    def test_unary_not(self):
        # both docs contain 'magnet' so NOT magnet -> empty
        self.assertQuery('NOT magnet', [])

    def test_or_not(self):
        self.assertQuery('magnet OR NOT superconductor', ['Doc1','Doc2'])

    def test_wildcard_prefix(self):
        # s* should match several terms in both docs, so expect both docs
        self.assertQuery('s*', ['Doc1','Doc2'])

    def test_wildcard_and(self):
        # super* matches superconductor(s) in both docs, AND magnet -> both docs
        self.assertQuery('super* AND magnet', ['Doc1','Doc2'])


if __name__ == '__main__':
    unittest.main()
