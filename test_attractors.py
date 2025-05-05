import unittest

import numpy as np

from attractors import attractor_or_transient


class TestAttractorOrTransient(unittest.TestCase):

    def test_square_matrix(self):
        T_inf = np.array([[0.5, 0.5], [0.5, 0.5]])
        result = attractor_or_transient(T_inf)
        expected_result = np.array([[1], [1]])
        self.assertTrue(np.allclose(result, expected_result))

    def test_non_square_matrix(self):
        T_inf = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        with self.assertRaises(ValueError):
            attractor_or_transient(T_inf, DEBUG=True)

    def test_attractor_states(self):
        T_inf = np.array([[1, 0], [0, 1]])
        result = attractor_or_transient(T_inf)
        expected_result = np.array([[1], [1]])
        self.assertTrue(np.allclose(result, expected_result))

    def test_with_transient_state(self):
        T_inf = np.array([[0, 1], [0, 1]])
        result = attractor_or_transient(T_inf)
        expected_result = np.array([[0], [1]])
        self.assertTrue(np.allclose(result, expected_result))

    def test_debug_true(self):
        T_inf = np.array([[1, 0], [0, 1]])
        result = attractor_or_transient(T_inf, DEBUG=True)
        expected_result = np.array([[1], [1]])
        self.assertTrue(np.allclose(result, expected_result))


if __name__ == '__main__':
    unittest.main()