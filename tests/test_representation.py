import unittest

import numpy as np

from sdmarkov.representation import (
    states_to_indices,
    indices_to_states,
    partial_assignment_to_mask,
    partial_assignments_to_masks,
    mask_to_partial_assignment,
    mask_value_to_pattern,
    subtract_subcube,
    decompose_subcubes,
    Subcube,
)


class TestStatesToIndices(unittest.TestCase):
    def test_valid_input(self):
        input_data = [['0100', '0101', '0110', '0111', '1101', '1111'], 
                      ['1001', '1011'], 
                      ['0000', '0001', '0010'], 
                      ['0011'], 
                      ['1100', '1110'], 
                      ['1000', '1010']]
        expected_output = [[4, 5, 6, 7, 13, 15], 
                           [9, 11], 
                           [0, 1, 2], 
                           [3], 
                           [12, 14], 
                           [8, 10]]
        self.assertEqual(states_to_indices(input_data), expected_output)

    def test_empty_input(self):
        input_data = []
        expected_output = []
        self.assertEqual(states_to_indices(input_data), expected_output)

    def test_single_element_input(self):
        input_data = [['0100']]
        expected_output = [[4]]
        self.assertEqual(states_to_indices(input_data), expected_output)

    def test_debug_mode_enabled(self):
        input_data = [['110', '001', '010'], ['111', '000']]
        expected_output = [[6, 1, 2], [7, 0]]
        self.assertEqual(states_to_indices(input_data, DEBUG=True), expected_output)

    def test_invalid_input_different_length(self):
        input_data = [['0100', '0101', '0110', '0111', '1101', '1111'], 
                      ['1001', '10111'], 
                      ['0000', '0001', '0010'], 
                      ['0011'], 
                      ['1100', '1110'], 
                      ['1000', '1010']]
        with self.assertRaises(ValueError):
            states_to_indices(input_data, DEBUG=True)

    def test_invalid_input_duplicates(self):
        input_data = [['0100', '0101', '0110', '0111', '1101', '1111'], 
                      ['1001', '1001'], 
                      ['0000', '0001', '0010'], 
                      ['0011'], 
                      ['1100', '1110'], 
                      ['1000', '1010']]
        with self.assertRaises(ValueError):
            states_to_indices(input_data, DEBUG=True)


class TestIndicesToStates(unittest.TestCase):
    def test_empty_input(self):
        input_data = []
        expected_output = []
        self.assertEqual(indices_to_states(input_data, 4), expected_output)
    
    def test_single_element_input(self):
        input_data = [[4]]
        expected_output = [['0100']]
        self.assertEqual(indices_to_states(input_data, 4), expected_output)

    def test_valid_input(self):
        input_data = [[4, 5, 6, 7, 13, 15], 
                      [9, 11], 
                      [0, 1, 2], 
                      [3], 
                      [12, 14], 
                      [8, 10]]
        expected_output = [['0100', '0101', '0110', '0111', '1101', '1111'], 
                           ['1001', '1011'], 
                           ['0000', '0001', '0010'], 
                           ['0011'], 
                           ['1100', '1110'], 
                           ['1000', '1010']]
        self.assertEqual(indices_to_states(input_data, 4), expected_output)

    def test_debug_mode_enabled(self):
        input_data = [[6, 1, 2], [7, 0]]
        expected_output = [['110', '001', '010'], ['111', '000']]
        self.assertEqual(indices_to_states(input_data, 3, DEBUG=True), expected_output)

    def test_too_small_N(self):
        input_data = [[1, 2, 6], [0, 7]]
        with self.assertRaises(ValueError):
            indices_to_states(input_data, 2, DEBUG=True)

    def test_invalid_input_duplicates(self):
        input_data = [[1, 2, 6], [1, 7]]
        with self.assertRaises(ValueError):
            indices_to_states(input_data, 3, DEBUG=True)


class TestPartialAssignmentToMaskExamples(unittest.TestCase):
    def test_empty_assignment(self):
        nodes = ["A", "B", "C"]

        assignment = {}

        mask, value = partial_assignment_to_mask(assignment, nodes)

        expected_mask = np.array([False, False, False])
        expected_value = np.array([False, False, False])

        np.testing.assert_array_equal(mask, expected_mask)
        np.testing.assert_array_equal(value, expected_value)

    def test_single_variable_assignment(self):
        nodes = ["A", "B", "C"]

        assignment = {"B": 1}

        mask, value = partial_assignment_to_mask(assignment, nodes)

        expected_mask = np.array([False, True, False])
        expected_value = np.array([False, True, False])

        np.testing.assert_array_equal(mask, expected_mask)
        np.testing.assert_array_equal(value, expected_value)

    def test_multiple_variable_assignment(self):
        nodes = ["A", "B", "C", "D"]

        assignment = {"A": 0, "C": 1}

        mask, value = partial_assignment_to_mask(assignment, nodes)

        expected_mask = np.array([True, False, True, False])
        expected_value = np.array([False, False, True, False])

        np.testing.assert_array_equal(mask, expected_mask)
        np.testing.assert_array_equal(value, expected_value)


class TestPartialAssignmentsToMasksExamples(unittest.TestCase):
    def test_batch_conversion(self):
        nodes = ["A", "B", "C"]

        assignments = [
            {},                 # unconstrained
            {"A": 1},            # single constraint
            {"B": 0, "C": 1},    # multiple constraints
        ]

        mask, value = partial_assignments_to_masks(assignments, nodes)

        expected_mask = np.array([
            [False, True,  False],
            [False, False, True ],
            [False, False, True ],
        ])

        expected_value = np.array([
            [False, True,  False],
            [False, False, False],
            [False, False, True ],
        ])

        np.testing.assert_array_equal(mask, expected_mask)
        np.testing.assert_array_equal(value, expected_value)


class TestMaskToPartialAssignment(unittest.TestCase):
    def test_basic_example(self):
        mask = np.array([1, 0, 1, 0], dtype=bool)
        value = np.array([1, 0, 0, 1], dtype=bool)
        nodes = ['A', 'B', 'C', 'D']

        pa = mask_to_partial_assignment(mask, value, nodes)
        expected = {'A': 1, 'C': 0}

        self.assertEqual(pa, expected)

    def test_default_node_names(self):
        mask = np.array([1, 0, 1], dtype=bool)
        value = np.array([0, 1, 1], dtype=bool)

        pa = mask_to_partial_assignment(mask, value)
        expected = {'0': 0, '2': 1}

        self.assertEqual(pa, expected)

    def test_empty_mask(self):
        mask = np.array([0, 0], dtype=bool)
        value = np.array([1, 0], dtype=bool)

        pa = mask_to_partial_assignment(mask, value)
        self.assertEqual(pa, {})  # No constrained bits


class TestMaskValueToPattern(unittest.TestCase):
    def test_basic_patterns(self):
        mask = np.array([1, 0, 1, 0], dtype=bool)
        value = np.array([0, 0, 1, 0], dtype=bool)

        self.assertEqual(mask_value_to_pattern(mask, value), "0*1*")

    def test_all_free(self):
        mask = np.array([0, 0, 0], dtype=bool)
        value = np.array([0, 1, 0], dtype=bool)  # ignored

        self.assertEqual(mask_value_to_pattern(mask, value), "***")

    def test_all_fixed(self):
        mask = np.array([1, 1, 1], dtype=bool)
        value = np.array([1, 0, 1], dtype=bool)

        self.assertEqual(mask_value_to_pattern(mask, value), "101")


class TestSubtractSubcubeExamples(unittest.TestCase):
    def test_no_overlap(self):
        nodes = ["A", "B", "C"]

        A = partial_assignment_to_mask({"A": 0}, nodes)   # 0**
        B = partial_assignment_to_mask({"A": 1}, nodes)   # 1**

        res = subtract_subcube(A, B)

        patterns = [mask_value_to_pattern(m, v) for m, v in res]
        self.assertEqual(patterns, ["0**"])

    def test_full_coverage(self):
        nodes = ["A", "B"]

        A = partial_assignment_to_mask({"A": 1}, nodes)   # 1*
        B = partial_assignment_to_mask({"A": 1}, nodes)   # 1*

        res = subtract_subcube(A, B)

        self.assertEqual(res, [])

    def test_simple_partial_overlap(self):
        nodes = ["A", "B"]

        A = partial_assignment_to_mask({}, nodes)         # **
        B = partial_assignment_to_mask({"A": 0}, nodes)   # 0*

        res = subtract_subcube(A, B)

        patterns = sorted(
            mask_value_to_pattern(m, v) for m, v in res
        )

        self.assertEqual(patterns, ["1*"])

    def test_given_example(self):
        """
        A: 1*0*
        B: 10*1

        A \\ B:
          110*
          1000
        """
        nodes = ["A", "B", "C", "D"]

        A = partial_assignment_to_mask(
            {"A": 1, "C": 0},
            nodes,
        )  # 1*0*

        B = partial_assignment_to_mask(
            {"A": 1, "B": 0, "D": 1},
            nodes,
        )  # 10*1

        res = subtract_subcube(A, B)

        patterns = sorted(
            mask_value_to_pattern(m, v) for m, v in res
        )

        self.assertEqual(patterns, ["1000", "110*"])


class TestDecomposeSubcubes(unittest.TestCase):
    def _mask_value(self, pattern: str):
        """
        Helper: convert string pattern like '1*0*' to (mask, value)
        """
        mask = np.array([c != '*' for c in pattern], dtype=bool)
        value = np.array([c == '1' for c in pattern], dtype=bool)
        return mask, value

    def _canonical_patterns(self, subcubes: list[Subcube]):
        """
        Helper: return human-readable patterns from canonical subcubes
        """
        return [mask_value_to_pattern(m, v) for m, v, _ in subcubes]

    def test_simple_no_overlap(self):
        # A = 1*0*, B = 0*1*
        subcubes = [
            (*self._mask_value('1*0*'), 'A'),
            (*self._mask_value('0*1*'), 'B'),
        ]
        canonical = decompose_subcubes(subcubes)
        patterns = self._canonical_patterns(canonical)
        self.assertCountEqual(patterns, ['1*0*', '0*1*'])

    def test_simple_overlap(self):
        # A = 1*0*, B = 10*1
        subcubes = [
            (*self._mask_value('1*0*'), 'A'),
            (*self._mask_value('10*1'), 'B'),
            (*self._mask_value('1001'), 'C'),
        ]
        canonical = decompose_subcubes(subcubes)
        patterns = self._canonical_patterns(canonical)
        # Expected: A\B produces 110* and 1000
        self.assertCountEqual(patterns, ['110*', '1000', '1011', '1001'])

    def test_full_overlap(self):
        # A = 1*0*, B = 1*0*
        subcubes = [
            (*self._mask_value('1*0*'), 'A'),
            (*self._mask_value('1*0*'), 'B'),
        ]
        canonical = decompose_subcubes(subcubes)
        patterns = self._canonical_patterns(canonical)
        # They are identical, the first dominates, second gets subtracted
        self.assertCountEqual(patterns, ['1*0*'])

    def test_nested_subcubes(self):
        # A = ***, B = 1*0
        subcubes = [
            (*self._mask_value('***'), 'A'),
            (*self._mask_value('1*0'), 'B'),
        ]
        canonical = decompose_subcubes(subcubes)
        patterns = self._canonical_patterns(canonical)
        # B is fully inside A, so A\B = 0**, 1*1
        self.assertIn('1*0', patterns)
        self.assertIn('0**', patterns)
        self.assertIn('1*1', patterns)


if __name__ == '__main__':
    unittest.main()