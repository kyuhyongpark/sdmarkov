import unittest

import numpy as np

from reachability import get_convergence_matrix


class TestGetConvergenceMatrix(unittest.TestCase):
    def test_valid_input_debug_false(self):
        T_inf = np.array([[0, 1], [0, 1]])
        scc_indices = [[0], [1]]
        attractor_indices = [[1]]
        expected_convergence_matrix = np.array([[1]])
        expected_transient_states = ['0']
        expected_attractor_states = [['1']]
        convergence_matrix, transient_states, attractor_states = get_convergence_matrix(T_inf, scc_indices, attractor_indices)
        self.assertEqual(convergence_matrix, expected_convergence_matrix)
        self.assertEqual(transient_states, expected_transient_states)
        self.assertEqual(attractor_states, expected_attractor_states)

    def test_valid_input_debug_true(self):
        T_inf = np.array([[0, 1], [0, 1]])
        scc_indices = [[0], [1]]
        attractor_indices = [[1]]
        expected_convergence_matrix = np.array([[1]])
        expected_transient_states = ['0']
        expected_attractor_states = [['1']]
        convergence_matrix, transient_states, attractor_states = get_convergence_matrix(T_inf, scc_indices, attractor_indices, DEBUG=True)
        self.assertEqual(convergence_matrix, expected_convergence_matrix)
        self.assertEqual(transient_states, expected_transient_states)
        self.assertEqual(attractor_states, expected_attractor_states)

    def test_invalid_scc_indices_small(self):
        T_inf = np.array([[0, 1], [0, 1]])
        scc_indices = [[1]]  # too small
        attractor_indices = [[1]]
        with self.assertRaises(ValueError):
            get_convergence_matrix(T_inf, scc_indices, attractor_indices, DEBUG=True)

    def test_invalid_scc_indices_out_of_range(self):
        T_inf = np.array([[0, 1], [0, 1]])
        scc_indices = [[0], [2]]  # out of range
        attractor_indices = [[1]]
        with self.assertRaises(ValueError):
            get_convergence_matrix(T_inf, scc_indices, attractor_indices, DEBUG=True)

    def test_invalid_attractor_indices_out_of_range(self):
        T_inf = np.array([[0, 1], [0, 1]])
        scc_indices = [[0], [1]]
        attractor_indices = [[2]]  # out of range
        with self.assertRaises(ValueError):
            get_convergence_matrix(T_inf, scc_indices, attractor_indices, DEBUG=True)

    def test_invalid_scc_indices_not_mutually_exclusive(self):
        T_inf = np.array([[0, 1], [0, 1]])
        scc_indices = [[0, 1], [1]]  # not mutually exclusive
        attractor_indices = [[1]]
        with self.assertRaises(ValueError):
            get_convergence_matrix(T_inf, scc_indices, attractor_indices, DEBUG=True)

    def test_invalid_attractor_indices_not_mutually_exclusive(self):
        T_inf = np.array([[0, 1], [0, 1]])
        scc_indices = [[0], [1]]
        attractor_indices = [[0], [0]]  # not mutually exclusive
        with self.assertRaises(ValueError):
            get_convergence_matrix(T_inf, scc_indices, attractor_indices, DEBUG=True)

    def test_invalid_transition_matrix(self):
        T_inf = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])  # non-square
        scc_indices = [[0], [1]]
        attractor_indices = [[0]]
        with self.assertRaises(ValueError):
            get_convergence_matrix(T_inf, scc_indices, attractor_indices, DEBUG=True)

    def test_empty_attractor_indices(self):
        T_inf = np.array([[0, 1], [0, 1]])
        scc_indices = [[0], [1]]
        attractor_indices = []
        with self.assertRaises(ValueError):
            get_convergence_matrix(T_inf, scc_indices, attractor_indices, DEBUG=True)

    def test_attractor_not_in_scc(self):
        T_inf = np.array([[0.5, 0.5], [0.5, 0.5]])
        scc_indices = [[0, 1]]
        attractor_indices = [[1]]
        with self.assertRaises(ValueError):
            get_convergence_matrix(T_inf, scc_indices, attractor_indices, DEBUG=True)


if __name__ == '__main__':
    unittest.main()