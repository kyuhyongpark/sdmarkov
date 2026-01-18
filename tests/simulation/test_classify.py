import unittest

import numpy as np

from sdmarkov.simulation.classify import classify_walkers
from sdmarkov.simulation.sample import prepare_group_sampler


# Small helper to create a test sampler
def make_test_sampler(xp):
    canonical_subcubes = [
        (np.array([1, 0, 0], bool), np.array([0, 0, 0], bool), "A"),  # 0**
        (np.array([1, 1, 0], bool), np.array([1, 0, 0], bool), "A"),  # 10*
        (np.array([1, 1, 0], bool), np.array([1, 1, 0], bool), "B"),  # 11*
    ]
    return prepare_group_sampler(canonical_subcubes, xp=xp)


class TestClassifyWalkers(unittest.TestCase):
    def test_single_walker(self):
        sampler = make_test_sampler(np)
        # Single walker in group A
        states = np.array([[0], [0], [0]], bool)  # shape (N=3, W=1)
        groups = classify_walkers(states, sampler, np)
        self.assertEqual(groups[0], sampler.group_name_to_id["A"])

    def test_multiple_walkers(self):
        sampler = make_test_sampler(np)
        # Two walkers: one in group A, one in group B
        states = np.array([
            [0, 1],
            [0, 1],
            [0, 1],
        ], bool)  # shape (N=3, W=2)
        groups = classify_walkers(states, sampler, np)
        self.assertEqual(groups[0], sampler.group_name_to_id["A"])
        self.assertEqual(groups[1], sampler.group_name_to_id["B"])

    def test_all_walkers_in_group_B(self):
        sampler = make_test_sampler(np)
        # Three walkers all in group B
        states = np.ones((3, 3), bool)  # shape (3,3)
        groups = classify_walkers(states, sampler, np)
        expected_id = sampler.group_name_to_id["B"]
        for g in groups:
            self.assertEqual(g, expected_id)


if __name__ == "__main__":
    unittest.main()
