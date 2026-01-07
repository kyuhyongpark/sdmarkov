import unittest

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from typing import List, Tuple

# Type aliases
Mask = np.ndarray  # shape (N,), bool
Value = np.ndarray  # shape (N,), bool
Subcube = Tuple[Mask, Value, str]

from sdmarkov.representation import (
    partial_assignment_to_mask,
    partial_assignments_to_masks,
    subtract_subcube,
    decompose_subcubes,
)
from sdmarkov.succession_diagram import close_trap_spaces_under_intersection

# ---------------- Strategies ----------------
@st.composite
def nodes_and_partial_assignments(draw):
    n = draw(st.integers(min_value=1, max_value=5))
    nodes = [f"X{i}" for i in range(n)]

    assignment = draw(
        st.dictionaries(
            keys=st.sampled_from(nodes),
            values=st.integers(min_value=0, max_value=1),
            max_size=n,
        )
    )

    return nodes, assignment

@st.composite
def nodes_and_multiple_partial_assignments(draw):
    nodes, first = draw(nodes_and_partial_assignments())

    assignments = draw(
        st.lists(
            st.dictionaries(
                keys=st.sampled_from(nodes),
                values=st.integers(min_value=0, max_value=1),
                max_size=len(nodes),
            ),
            min_size=1,
            max_size=6,
        )
    )

    return nodes, assignments


@st.composite
def subcube(draw, max_n=5):
    N = draw(st.integers(min_value=1, max_value=max_n))

    mask = draw(
        st.lists(
            st.booleans(),
            min_size=N,
            max_size=N,
        )
    )

    value = []
    for m in mask:
        if m:
            value.append(draw(st.booleans()))
        else:
            value.append(False)

    return (
        np.array(mask, dtype=bool),
        np.array(value, dtype=bool),
    )


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


# Helper functions
def states_satisfying(mask, value):
    """
    Enumerate all concrete states satisfying (state & mask) == value.
    Only suitable for small N.
    """
    N = len(mask)
    states = set()

    for i in range(2 ** N):
        bits = tuple((i >> k) & 1 for k in range(N))
        if all(
            (not mask[j]) or bits[j] == value[j]
            for j in range(N)
        ):
            states.add(bits)

    return states


# ---------------- Property Test ----------------
class TestPartialAssignmentToMaskProperties(unittest.TestCase):
    @given(nodes_and_partial_assignments())
    def test_mask_value_invariants(self, data):
        nodes, assignment = data

        mask, value = partial_assignment_to_mask(assignment, nodes)

        N = len(nodes)
        self.assertEqual(mask.shape, (N,))
        self.assertEqual(value.shape, (N,))

        for i, node in enumerate(nodes):
            if node in assignment:
                self.assertTrue(mask[i])
                self.assertEqual(value[i], bool(assignment[node]))
            else:
                self.assertFalse(mask[i])
                self.assertFalse(value[i])


class TestPartialAssignmentsToMasksProperties(unittest.TestCase):
    @given(nodes_and_multiple_partial_assignments())
    def test_batch_equals_scalar_stacking(self, data):
        nodes, assignments = data

        batch_mask, batch_value = partial_assignments_to_masks(
            assignments, nodes
        )

        scalar_masks = []
        scalar_values = []

        for assignment in assignments:
            m, v = partial_assignment_to_mask(assignment, nodes)
            scalar_masks.append(m)
            scalar_values.append(v)

        expected_mask = np.stack(scalar_masks, axis=1)
        expected_value = np.stack(scalar_values, axis=1)

        np.testing.assert_array_equal(batch_mask, expected_mask)
        np.testing.assert_array_equal(batch_value, expected_value)


class TestSubtractSubcubeProperties(unittest.TestCase):
    @given(A=subcube(), B=subcube())
    def test_subtract_subcube_correctness(self, A, B):
        mA, vA = A
        mB, vB = B

        # Only compare subcubes of the same dimension
        if len(mA) != len(mB):
            return

        residuals = subtract_subcube(A, B)

        A_states = states_satisfying(mA, vA)
        B_states = states_satisfying(mB, vB)

        expected = A_states - B_states

        actual = set()
        for m, v in residuals:
            actual |= states_satisfying(m, v)

        assert actual == expected

    @given(A=subcube(), B=subcube())
    def test_subtract_subcube_disjointness(self, A, B):
        mA, vA = A
        mB, vB = B

        if len(mA) != len(mB):
            return

        residuals = subtract_subcube(A, B)

        state_sets = [
            states_satisfying(m, v)
            for m, v in residuals
        ]

        for i in range(len(state_sets)):
            for j in range(i + 1, len(state_sets)):
                assert state_sets[i].isdisjoint(state_sets[j])

    @given(A=subcube(), B=subcube())
    def test_subtract_subcube_no_overlap_identity(self, A, B):
        mA, vA = A
        mB, vB = B

        if len(mA) != len(mB):
            return

        A_states = states_satisfying(mA, vA)
        B_states = states_satisfying(mB, vB)

        if A_states.isdisjoint(B_states):
            res = subtract_subcube(A, B)
            assert len(res) == 1
            m, v = res[0]
            assert states_satisfying(m, v) == A_states

    @given(A=subcube(), B=subcube())
    def test_subtract_subcube_full_coverage_empty(self, A, B):
        mA, vA = A
        mB, vB = B

        if len(mA) != len(mB):
            return

        A_states = states_satisfying(mA, vA)
        B_states = states_satisfying(mB, vB)

        if A_states.issubset(B_states):
            res = subtract_subcube(A, B)
            assert res == []


class TestDecomposeSubcubesProperties(unittest.TestCase):
    @given(nodes_and_closed_trap_spaces())
    def test_disjoint_output(self, data):
        nodes, trap_spaces = data
        masks, values = partial_assignments_to_masks(trap_spaces, nodes)
        subcubes = [(masks[:, i], values[:, i], i) for i in range(masks.shape[1])]
        canonical = decompose_subcubes(subcubes, DEBUG=True)

        # Check disjointness by enumerating all states
        seen = set()
        for m, v, _ in canonical:
            for state in states_satisfying(m, v):
                key = tuple(state)
                self.assertNotIn(key, seen, "States must be disjoint across canonical subcubes")
                seen.add(key)

    @given(nodes_and_closed_trap_spaces())
    def test_full_coverage(self, data):
        nodes, trap_spaces = data
        masks, values = partial_assignments_to_masks(trap_spaces, nodes)
        subcubes = [(masks[:, i], values[:, i], i) for i in range(masks.shape[1])]
        canonical = decompose_subcubes(subcubes, DEBUG=True)

        # Check that all states in the original subcubes appear in canonical
        for m, v in zip(masks.T, values.T):
            for state in states_satisfying(m, v):
                covered = any(np.all((state & cm) == cv) for cm, cv, _ in canonical)
                self.assertTrue(covered, "Original subcube states must appear in canonical decomposition")

    @given(nodes_and_closed_trap_spaces())
    def test_original_subcubes_preserved(self, data):
        nodes, trap_spaces = data
        masks, values = partial_assignments_to_masks(trap_spaces, nodes)
        subcubes = [(masks[:, i], values[:, i], i) for i in range(masks.shape[1])]
        canonical = decompose_subcubes(subcubes, DEBUG=True)

        for cm, cv, group in canonical:
            m_orig, v_orig = subcubes[group][:2]

            # Only check if the original subcube has constrained bits
            if not m_orig.any():
                continue

            # Enumerate all concrete states in the canonical subcube
            for state in states_satisfying(cm, cv):
                # For each state, verify it satisfies the original subcube constraints
                for idx, bit in enumerate(m_orig):
                    if bit:  # only check constrained positions
                        assert state[idx] == v_orig[idx], (
                            "Canonical subcube must be contained in its original subcube"
                        )

    @given(nodes_and_closed_trap_spaces())
    def test_order_invariance(self, data):
        """
        Changing the order of intersection-closed input subcubes
        must not affect the canonical decomposition.
        """
        nodes, trap_spaces = data
        masks, values = partial_assignments_to_masks(trap_spaces, nodes)
        subcubes = [(masks[:, i], values[:, i], i) for i in range(masks.shape[1])]

        out1 = decompose_subcubes(subcubes)

        shuffled = subcubes[:]
        np.random.shuffle(shuffled)
        out2 = decompose_subcubes(shuffled)

        def normalize(output):
            """
            Convert output to a set of hashable, order-independent items.
            """
            return {
                (
                    tuple(mask.tolist()),
                    tuple(value.tolist()),
                    group_id
                )
                for mask, value, group_id in output
            }

        self.assertEqual(
            normalize(out1),
            normalize(out2),
            "Canonical decomposition depends on input order"
        )


if __name__ == '__main__':
    unittest.main()