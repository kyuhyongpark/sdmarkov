import unittest
import numpy as np
import cupy as cp

from sdmarkov.simulation.sample import (
    prepare_group_sampler,
    sample_walkers_from_group,
)

class TestPrepareGroupSampler(unittest.TestCase):
    def test_basic_translation(self):
        xp = np

        # canonical decomposition
        # A = {00, 01, 10}, B = {11}
        canonical_subcubes = [
            (np.array([1, 0], bool), np.array([0, 0], bool), "A"),  # 0*
            (np.array([1, 1], bool), np.array([1, 0], bool), "A"),  # 10
            (np.array([1, 1], bool), np.array([1, 1], bool), "B"),  # 11
        ]
        sampler = prepare_group_sampler(canonical_subcubes, xp=xp)

        # shapes
        self.assertEqual(sampler.masks.shape, (2, 3))
        self.assertEqual(sampler.values.shape, (2, 3))
        self.assertEqual(sampler.group_ids.shape, (3,))

        # free bits
        self.assertListEqual(sampler.free_bits.tolist(), [1, 0, 0])

        # group structure
        groups = sampler.groups
        self.assertEqual(len(groups), 2)

        sizes = sorted(len(ks) for ks in groups.values())
        self.assertListEqual(sizes, [1, 2])

        # group log sizes consistency
        for g, ks in groups.items():
            lhs = np.exp(sampler.group_log_sizes[g])
            rhs = np.exp(sampler.log_sizes[ks]).sum()
            self.assertTrue(np.isclose(lhs, rhs))

        self.assertEqual(sampler.group_name_to_id, {"A": 0, "B": 1})
        self.assertEqual(sampler.id_to_group_name, ["A", "B"])


class TestSampleWalkersFromGroup(unittest.TestCase):
    def setUp(self):
        xp = np
        canonical_subcubes = [
            (np.array([1, 0], bool), np.array([0, 0], bool), "A"),  # 0*
            (np.array([1, 1], bool), np.array([1, 0], bool), "A"),  # 10
            (np.array([1, 1], bool), np.array([1, 1], bool), "B"),  # 11
        ]
        self.sampler = prepare_group_sampler(canonical_subcubes, xp=xp)
        self.xp = xp

    def test_samples_are_in_group_A(self):
        xp = self.xp
        W = 2000

        states, subcube_idx = sample_walkers_from_group(self.sampler, target_group="A", n_walkers=W, xp=xp)

        # valid states for group A are {00, 01, 10}
        allowed = {(0,0), (0,1), (1,0)}

        for i in range(W):
            s = tuple(int(x) for x in states[:, i])
            self.assertIn(s, allowed)

    def test_samples_are_in_group_B(self):
        xp = self.xp
        W = 500

        states, subcube_idx = sample_walkers_from_group(self.sampler, target_group="B", n_walkers=W, xp=xp)

        for i in range(W):
            self.assertEqual(tuple(states[:, i].tolist()), (1,1))


class TestUsingCupy(unittest.TestCase):
    @unittest.skipIf(cp is None, "cupy not available")
    def test_numpy_cupy_parity(self):

        canonical_subcubes = [
            (np.array([1, 0], bool), np.array([0, 0], bool), "A"),  # 0*
            (np.array([1, 1], bool), np.array([1, 0], bool), "A"),  # 10
            (np.array([1, 1], bool), np.array([1, 1], bool), "B"),  # 11
        ]
                
        sampler_np = prepare_group_sampler(canonical_subcubes, xp=np)
        sampler_cp = prepare_group_sampler(canonical_subcubes, xp=cp)

        W = 10000

        states_np, _ = sample_walkers_from_group(sampler_np, "A", W, xp=np)
        states_cp, _ = sample_walkers_from_group(sampler_cp, "A", W, xp=cp)

        # move cupy result to cpu
        states_cp = cp.asnumpy(states_cp)

        # compare empirical distributions
        def histogram(states):
            return np.mean(states, axis=1)

        self.assertTrue(
            np.allclose(
                histogram(states_np),
                histogram(states_cp),
                atol=0.02,
            )
        )


if __name__ == "__main__":
    unittest.main()
