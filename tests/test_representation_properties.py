import unittest

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from sdmarkov.representation import (
    partial_assignment_to_mask,
    partial_assignments_to_masks,
)

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


if __name__ == '__main__':
    unittest.main()