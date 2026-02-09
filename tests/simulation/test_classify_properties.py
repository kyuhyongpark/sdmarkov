import unittest
import copy

import numpy as np
from hypothesis import given, strategies as st, settings

from sdmarkov.representation import (
    partial_assignments_to_masks,
    decompose_subcubes,
)
from sdmarkov.succession_diagram import close_trap_spaces_under_intersection
from sdmarkov.simulation.classify import classify_walkers
from sdmarkov.simulation.sample import (
    prepare_group_sampler
)

# ---------------- Assume necessary functions are imported ----------------
# prepare_group_sampler, classify_walkers, close_trap_spaces_under_intersection, 
# partial_assignments_to_masks, decompose_subcubes

# ---------------- Hypothesis strategies ----------------
@st.composite
def nodes_and_closed_trap_spaces(draw):
    # Number of nodes
    n = draw(st.integers(min_value=1, max_value=5))
    nodes = [f"X{i}" for i in range(n)]

    # Strategy for a single trap space (partial assignment)
    def trap_space_strategy():
        return st.dictionaries(
            keys=st.sampled_from(nodes),
            values=st.integers(min_value=0, max_value=1),
            max_size=n,
        )

    # Draw 1–5 random trap spaces
    raw = draw(
        st.lists(
            trap_space_strategy(),
            min_size=1,
            max_size=5,
            unique_by=lambda d: frozenset(d.items()),
        )
    )

    # Always include the empty trap space
    raw.append({})

    # Close under intersection
    closed = close_trap_spaces_under_intersection(raw)

    return nodes, closed

@st.composite
def canonical_decompositions(draw):
    nodes, closed = draw(nodes_and_closed_trap_spaces())

    masks, values = partial_assignments_to_masks(closed, nodes)

    # Give each trap space a group ID
    groups = list(range(len(closed)))
    subcubes = [(masks[:, i], values[:, i], groups[i]) for i in range(masks.shape[1])]

    canonical = decompose_subcubes(subcubes)

    return nodes, canonical

# ---------------- Property-based test class ----------------
class TestClassifyWalkersProperty(unittest.TestCase):
    @given(canonical_decompositions())
    @settings(max_examples=20)
    def test_classify_walkers_properties(self, data):
        nodes, canonical = data
        xp = np  # can switch to cp for GPU

        # Prepare sampler
        sampler = prepare_group_sampler(canonical, xp=xp)

        # Generate random states to test
        N = len(nodes)
        W = 10
        states = xp.random.randint(0, 2, size=(N, W), dtype=bool)

        # --- Classification ---
        group_ids, subcube_idx = classify_walkers(states, sampler, xp=xp)

        # --- Every walker is classified and belongs to exactly one group ---
        self.assertEqual(group_ids.shape, (W,))
        for gid in group_ids:
            self.assertIn(int(gid), sampler.groups)

        # --- Idempotence ---
        group_ids2, subcube_idx = classify_walkers(states, sampler, xp=xp)
        self.assertTrue(xp.all(group_ids == group_ids2))

        # --- Permutation invariance ---
        # Shuffle the canonical subcubes internally
        canonical_shuffled = copy.deepcopy(canonical)
        xp.random.shuffle(canonical_shuffled)
        sampler_shuffled = prepare_group_sampler(canonical_shuffled, xp=xp)

        group_ids_shuffled, subcube_idx = classify_walkers(states, sampler_shuffled, xp=xp)
        
        id_to_name = sampler.id_to_group_name
        group_names_original = [id_to_name[gid] for gid in group_ids]

        id_to_name_shuffled = sampler_shuffled.id_to_group_name
        group_names_shuffled = [id_to_name_shuffled[gid] for gid in group_ids_shuffled]

        # Now compare the names
        assert np.all(np.array(group_names_original) == np.array(group_names_shuffled))


if __name__ == "__main__":
    unittest.main()
