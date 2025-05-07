import unittest

import numpy as np
from pyboolnet.external.bnet2primes import bnet_text2primes
from pyboolnet.state_transition_graphs import primes2stg

from basins import get_convergence_matrix, get_strong_basins, get_basin_ratios
from transition_matrix import get_transition_matrix
from grouping import sd_grouping, null_grouping
from matrix_operations import nsquare, compress_matrix, expand_matrix
from attractors import get_predicted_attractors


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


# class TestGetBasinRatios(unittest.TestCase):
#     def test_simple_transition_matrix(self):
#         T_inf = np.array([[0, 1], [0, 1]])
#         attractor_indices = [[1]]
#         basin_ratios, attractor_states = get_basin_ratios(T_inf, attractor_indices)
#         expected_basin_ratios = np.array([[1]])
#         expected_attractor_states = [["1"]]
#         self.assertTrue(np.allclose(basin_ratios, expected_basin_ratios))
#         self.assertEqual(attractor_states, expected_attractor_states)

#     def test_with_debug(self):
#         T_inf = np.array([[0, 1], [0, 1]])
#         attractor_indices = [[1]]
#         basin_ratios, attractor_states = get_basin_ratios(T_inf, attractor_indices, DEBUG=True)
#         expected_basin_ratios = np.array([[1]])
#         expected_attractor_states = [["1"]]
#         self.assertTrue(np.allclose(basin_ratios, expected_basin_ratios))
#         self.assertEqual(attractor_states, expected_attractor_states)

#     def test_example(self):
#         bnet = """
#         A, A | B & C
#         B, B & !C
#         C, B & !C | !C & !D | !B & C & D
#         D, !A & !B & !C & !D | !A & C & D
#         """

#         update = "asynchronous"

#         primes = bnet_text2primes(bnet)
#         primes = {key: primes[key] for key in sorted(primes)}
#         stg = primes2stg(primes, update)
#         scc_dag = get_scc_dag(stg)
#         attractor_indices = get_attractor_states(scc_dag, as_indices=True, DEBUG=True)
#         transition_matrix = get_transition_matrix(stg, update=update, DEBUG=True)
#         T_inf = nsquare(transition_matrix, 20, DEBUG=True)

#         basin_ratios, attractor_states = get_basin_ratios(T_inf, attractor_indices, DEBUG=True)
#         expected_basin_ratios = np.array([[0.275, 0.1, 0.625]])
#         expected_attractor_states = [["0000", "0001", "0010"], ["0011"], ["1000", "1010"]]
#         self.assertTrue(np.allclose(basin_ratios, expected_basin_ratios))
#         self.assertEqual(attractor_states, expected_attractor_states)


if __name__ == '__main__':
    unittest.main()