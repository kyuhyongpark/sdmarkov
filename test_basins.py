import unittest

import numpy as np
from pyboolnet.external.bnet2primes import bnet_text2primes
from pyboolnet.state_transition_graphs import primes2stg

from attractors import get_predicted_attractors
from basins import get_convergence_matrix, get_strong_basins, get_basin_ratios, get_node_average_values
from grouping import sd_grouping, null_grouping
from matrix_operations import nsquare, compress_matrix, expand_matrix
from transition_matrix import get_transition_matrix


class TestGetConvergenceMatrix(unittest.TestCase):
    def test_valid_input_debug_false(self):
        T_inf = np.array([[0, 1], [0, 1]])
        attractor_indices = [[1]]
        expected_convergence_matrix = np.array([[1], [1]])
        convergence_matrix = get_convergence_matrix(T_inf, attractor_indices)
        self.assertTrue(np.allclose(convergence_matrix, expected_convergence_matrix))

    def test_valid_input_debug_true(self):
        T_inf = np.array([[0, 1], [0, 1]])
        attractor_indices = [[1]]
        expected_convergence_matrix = np.array([[1], [1]])
        convergence_matrix = get_convergence_matrix(T_inf, attractor_indices, DEBUG=True)
        self.assertTrue(np.allclose(convergence_matrix, expected_convergence_matrix))

    def test_invalid_attractor_indices_out_of_range(self):
        T_inf = np.array([[0, 1], [0, 1]])
        attractor_indices = [[2]]  # out of range
        with self.assertRaises(ValueError):
            get_convergence_matrix(T_inf, attractor_indices, DEBUG=True)

    def test_invalid_attractor_indices_not_mutually_exclusive(self):
        T_inf = np.array([[0, 1], [0, 1]])
        attractor_indices = [[0], [0]]  # not mutually exclusive
        with self.assertRaises(ValueError):
            get_convergence_matrix(T_inf, attractor_indices, DEBUG=True)

    def test_invalid_transition_matrix(self):
        T_inf = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])  # non-square
        attractor_indices = [[0]]
        with self.assertRaises(ValueError):
            get_convergence_matrix(T_inf, attractor_indices, DEBUG=True)

    def test_empty_attractor_indices(self):
        T_inf = np.array([[0, 1], [0, 1]])
        attractor_indices = []
        with self.assertRaises(ValueError):
            get_convergence_matrix(T_inf, attractor_indices, DEBUG=True)


class TestGetStrongBasins(unittest.TestCase):
    def test_simple_transition_matrix(self):
        convergence_matrix = np.array([[1, 0], [0, 1]])
        strong_basin = get_strong_basins(convergence_matrix)
        expected_strong_basin = np.array([[1], [1]])
        self.assertTrue(np.allclose(strong_basin, expected_strong_basin))

    def test_multiple_attractors(self):
        convergence_matrix = np.array([[1/2, 1/2], [0, 1]])
        strong_basin = get_strong_basins(convergence_matrix)
        expected_strong_basin = np.array([[0], [1]])
        self.assertTrue(np.allclose(strong_basin, expected_strong_basin))

    def test_example(self):
        bnet = """
        A, A | B & C
        B, B & !C
        C, B & !C | !C & !D | !B & C & D
        D, !A & !B & !C & !D | !A & C & D
        """

        update = "asynchronous"

        primes = bnet_text2primes(bnet)
        primes = {key: primes[key] for key in sorted(primes)}
        stg = primes2stg(primes, update)

        transition_matrix = get_transition_matrix(stg, update=update)
        T_inf = nsquare(transition_matrix, 20, DEBUG=True)
        attractor_indices = get_predicted_attractors(T_inf, as_indices=True, DEBUG=True)

        convergence_matrix = get_convergence_matrix(T_inf, attractor_indices, DEBUG=True)

        strong_basin = get_strong_basins(convergence_matrix, DEBUG=True)
        expected_strong_basin = np.array([[1],
                                          [1],
                                          [1],
                                          [1],
                                          [0],
                                          [0],
                                          [0],
                                          [0],
                                          [1],
                                          [1],
                                          [1],
                                          [1],
                                          [1],
                                          [1],
                                          [1],
                                          [1]])

        self.assertTrue(np.allclose(strong_basin, expected_strong_basin))

    def test_null_example(self):
        bnet = """
        A, A | B & C
        B, B & !C
        C, B & !C | !C & !D | !B & C & D
        D, !A & !B & !C & !D | !A & C & D
        """

        update = "asynchronous"

        primes = bnet_text2primes(bnet)
        primes = {key: primes[key] for key in sorted(primes)}
        stg = primes2stg(primes, update)

        transition_matrix = get_transition_matrix(stg, update=update)
        null_group = null_grouping(bnet)
        null_matrix = compress_matrix(transition_matrix, null_group)

        predicted_attractor_indices = get_predicted_attractors(null_matrix, null_group, as_indices=True, DEBUG=True)

        null_inf = nsquare(null_matrix, 20, DEBUG=True)
        null_inf_expanded = expand_matrix(null_inf, null_group)

        convergence_matrix = get_convergence_matrix(null_inf_expanded, predicted_attractor_indices, DEBUG=True)

        strong_basin = get_strong_basins(convergence_matrix, DEBUG=True)
        expected_strong_basin = np.array([[0],
                                          [0],
                                          [0],
                                          [1],
                                          [0],
                                          [0],
                                          [0],
                                          [0],
                                          [1],
                                          [0],
                                          [1],
                                          [0],
                                          [0],
                                          [0],
                                          [0],
                                          [0]])

        self.assertTrue(np.allclose(strong_basin, expected_strong_basin))

    def test_sd_example(self):
        bnet = """
        A, A | B & C
        B, B & !C
        C, B & !C | !C & !D | !B & C & D
        D, !A & !B & !C & !D | !A & C & D
        """

        update = "asynchronous"

        primes = bnet_text2primes(bnet)
        primes = {key: primes[key] for key in sorted(primes)}
        stg = primes2stg(primes, update)

        transition_matrix = get_transition_matrix(stg, update=update)
        sd_group = sd_grouping(bnet)
        sd_matrix = compress_matrix(transition_matrix, sd_group)

        predicted_attractor_indices = get_predicted_attractors(sd_matrix, sd_group, as_indices=True, DEBUG=True)

        sd_inf = nsquare(sd_matrix, 20, DEBUG=True)
        sd_inf_expanded = expand_matrix(sd_inf, sd_group)

        convergence_matrix = get_convergence_matrix(sd_inf_expanded, predicted_attractor_indices, DEBUG=True)

        strong_basin = get_strong_basins(convergence_matrix, DEBUG=True)
        expected_strong_basin = np.array([[1],
                                          [1],
                                          [1],
                                          [1],
                                          [0],
                                          [0],
                                          [0],
                                          [0],
                                          [1],
                                          [1],
                                          [1],
                                          [1],
                                          [1],
                                          [1],
                                          [1],
                                          [1]])

        self.assertTrue(np.allclose(strong_basin, expected_strong_basin))


class TestGetBasinRatios(unittest.TestCase):
    def test_simple_transition_matrix(self):
        convergence_matrix = np.array([[0, 1], [0, 1]])
        basin_ratios = get_basin_ratios(convergence_matrix)
        expected_basin_ratios = np.array([[0, 1]])
        self.assertTrue(np.allclose(basin_ratios, expected_basin_ratios))

    def test_with_debug(self):
        convergence_matrix = np.array([[0, 1], [0, 1]])
        basin_ratios = get_basin_ratios(convergence_matrix, DEBUG=True)
        expected_basin_ratios = np.array([[0, 1]])
        self.assertTrue(np.allclose(basin_ratios, expected_basin_ratios))

    def test_example(self):
        bnet = """
        A, A | B & C
        B, B & !C
        C, B & !C | !C & !D | !B & C & D
        D, !A & !B & !C & !D | !A & C & D
        """

        update = "asynchronous"

        primes = bnet_text2primes(bnet)
        primes = {key: primes[key] for key in sorted(primes)}
        stg = primes2stg(primes, update)

        transition_matrix = get_transition_matrix(stg, update=update, DEBUG=True)
        T_inf = nsquare(transition_matrix, 20, DEBUG=True)

        attractor_indices = get_predicted_attractors(transition_matrix, as_indices=True, DEBUG=True)
        convergence_matrix = get_convergence_matrix(T_inf, attractor_indices, DEBUG=True)

        basin_ratios = get_basin_ratios(convergence_matrix, DEBUG=True)
        expected_basin_ratios = np.array([[0.275, 0.1, 0.625]])

        self.assertTrue(np.allclose(basin_ratios, expected_basin_ratios))


class TestGetNodeAverageValues(unittest.TestCase):
    def test_2x2_transition_matrix(self):
        T_inf_expanded = np.array([[0.5, 0.5], [0.5, 0.5]])
        expected_result = np.array([[0.5]])
        self.assertTrue(np.allclose(get_node_average_values(T_inf_expanded), expected_result))

    def test_2x2_transition_matrix2(self):
        T_inf_expanded = np.array([[1, 0], [0, 1]])
        expected_result = np.array([[0.5]])
        self.assertTrue(np.allclose(get_node_average_values(T_inf_expanded), expected_result))

    def test_2x2_transition_matrix2(self):
        T_inf_expanded = np.array([[0, 1], [0, 1]])
        expected_result = np.array([[1]])
        self.assertTrue(np.allclose(get_node_average_values(T_inf_expanded), expected_result))

    def test_4x4_transition_matrix(self):
        T_inf_expanded = np.array([[0.25, 0.25, 0.25, 0.25], 
                                   [0.25, 0.25, 0.25, 0.25], 
                                   [0.25, 0.25, 0.25, 0.25], 
                                   [0.25, 0.25, 0.25, 0.25]])
        expected_result = np.array([[0.5, 0.5]])
        self.assertTrue(np.allclose(get_node_average_values(T_inf_expanded), expected_result))

    def test_4x4_transition_matrix2(self):
        T_inf_expanded = np.array([[1, 0, 0, 0], 
                                   [0, 1, 0, 0], 
                                   [0, 0, 1, 0], 
                                   [0, 0, 0, 1]])
        expected_result = np.array([[0.5, 0.5]])
        self.assertTrue(np.allclose(get_node_average_values(T_inf_expanded), expected_result))

    def test_non_square_transition_matrix(self):
        T_inf_expanded = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        with self.assertRaises(ValueError):
            get_node_average_values(T_inf_expanded, DEBUG=True)

    def test_invalid_transition_matrix(self):
        T_inf_expanded = np.array([[0.5, 0.5], [0.5, 1.5]])
        with self.assertRaises(ValueError):
            get_node_average_values(T_inf_expanded, DEBUG=True)

    def test_debug_true(self):
        T_inf_expanded = np.array([[0.5, 0.5], [0.5, 0.5]])
        expected_result = np.array([[0.5]])
        self.assertTrue(np.allclose(get_node_average_values(T_inf_expanded, DEBUG=True), expected_result))

    def test_example(self):
        bnet = """
        A, A | B & C
        B, B & !C
        C, B & !C | !C & !D | !B & C & D
        D, !A & !B & !C & !D | !A & C & D
        """

        update = "asynchronous"

        DEBUG = True

        primes = bnet_text2primes(bnet)
        primes = {key: primes[key] for key in sorted(primes)}
        stg = primes2stg(primes, update)

        T = get_transition_matrix(stg, update=update, DEBUG=DEBUG)
        T_inf = nsquare(T, 20, DEBUG=DEBUG)

        T_node_average_values = get_node_average_values(T_inf, DEBUG=DEBUG)

        expected_T_node_average_values = np.array([[0.625, 0, 0.504, 0.192]])

        self.assertTrue(np.allclose(T_node_average_values, expected_T_node_average_values, atol=1e-3))


if __name__ == '__main__':
    unittest.main()