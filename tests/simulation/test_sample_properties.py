import unittest
import collections

import numpy as np
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from sdmarkov.representation import (
    partial_assignments_to_masks,
    decompose_subcubes,
)
from sdmarkov.succession_diagram import close_trap_spaces_under_intersection
from sdmarkov.simulation.sample import (
    prepare_group_sampler,
    sample_walkers_from_group,
)


# ---------------- Strategies ----------------
@st.composite
def nodes_and_closed_trap_spaces(draw):
    n = draw(st.integers(min_value=1, max_value=5))
    nodes = [f"X{i}" for i in range(n)]

    def trap_space_strategy():
        return st.dictionaries(
            keys=st.sampled_from(nodes),
            values=st.integers(min_value=0, max_value=1),
            max_size=n,
        )

    raw = draw(
        st.lists(
            trap_space_strategy(),
            min_size=1,
            max_size=6,
            unique_by=lambda d: frozenset(d.items()),
        )
    )

    # Close under intersection
    closed = close_trap_spaces_under_intersection(raw)

    return nodes, closed

@st.composite
def canonical_decompositions(draw):
    nodes, closed = draw(nodes_and_closed_trap_spaces())

    masks, values = partial_assignments_to_masks(closed, nodes)

    # give each trap space a group name
    groups = list(range(len(closed)))
    subcubes = [(masks[:, i], values[:, i], groups[i]) for i in range(masks.shape[1])]

    canonical = decompose_subcubes(subcubes)

    return nodes, canonical


# ---------------- Property Test ----------------
class TestPrepareGroupSamplerProperties(unittest.TestCase):
    @given(canonical_decompositions())
    def test_group_volume_consistency(self, data):
        nodes, canonical = data
        N = len(nodes)

        sampler = prepare_group_sampler(canonical, xp=np)

        for g, ks in sampler.groups.items():
            lhs = np.exp(sampler.group_log_sizes[g])
            rhs = np.exp(sampler.log_sizes[ks]).sum()
            self.assertTrue(np.isclose(lhs, rhs))

        # Subcube preservation
        for k in range(len(canonical)):
            self.assertEqual(
                sampler.free_bits[k],
                N - np.count_nonzero(sampler.masks[:, k])
            )

            self.assertTrue(
                np.isclose(
                    sampler.log_sizes[k],
                    sampler.free_bits[k] * np.log(2)
                )
            )

        # Group partition correctness
        all_indices = sorted(i for ks in sampler.groups.values() for i in ks)
        self.assertEqual(all_indices, list(range(len(canonical))))

        for g1, ks1 in sampler.groups.items():
            for g2, ks2 in sampler.groups.items():
                if g1 != g2:
                    self.assertTrue(set(ks1).isdisjoint(set(ks2)))


class TestSampleWalkersFromGroupProperties(unittest.TestCase):
    @settings(deadline=None, max_examples=30)
    @given(canonical_decompositions())
    def test_sampling_is_state_uniform(self, data):
        nodes, canonical = data

        sampler = prepare_group_sampler(canonical, xp=np)

        # pick any group
        g = list(sampler.groups.keys())[0]

        N = len(nodes)
        W = 2000 * max(1, 2**N)
        W = min(W, 20000)
        states = sample_walkers_from_group(sampler, g, W, xp=np)

        # ----- Semantic validity -----
        for i in range(W):
            matches = 0
            for k in sampler.groups[g]:
                m, v, _ = canonical[k]
                if np.all((states[:, i] & m) == v):
                    matches += 1
            self.assertEqual(matches, 1)

        # ----- Subcube selection distribution -----
        ks = sampler.groups[g]
        if len(ks) == 1 and sampler.free_bits[ks[0]] == 0:
            return

        # classify each sample to a canonical subcube
        def find_subcube(x):
            for k, (m, v, _) in enumerate(canonical):
                if np.all((x & m) == v):
                    return k
            raise RuntimeError("no match")

        hits = [find_subcube(states[:, i]) for i in range(W)]

        # theoretical weights
        log_sizes = sampler.log_sizes[ks]
        Z = np.exp(log_sizes).sum()
        probs = {k: np.exp(sampler.log_sizes[k]) / Z for k in ks}

        # empirical
        counts = collections.Counter(hits)

        for k in ks:
            emp = counts[k] / W
            self.assertAlmostEqual(emp, probs[k], delta=0.05)

        # ----- Uniformity inside subcube -----
        def free_bits_of(k, sampler):
            # returns array of bit indices i such that bit i is free in subcube k
            return np.where(~sampler.masks[:, k])[0]

        for k in ks:
            free = free_bits_of(k, sampler)
            samples = states[:, hits == k]

            for i in free:
                p = samples[i].mean()
                self.assertAlmostEqual(p, 0.5, delta=0.05)


if __name__ == "__main__":
    unittest.main()