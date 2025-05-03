import unittest

import numpy as np
from pyboolnet.external.bnet2primes import bnet_text2primes
from pyboolnet.state_transition_graphs import primes2stg

from basins import get_strong_basins, get_basin_ratios
from transition_matrix import get_transition_matrix
from grouping import sd_grouping, null_grouping
from matrix_operations import nsquare, compress_matrix, expand_matrix
from reachability import get_convergence_matrix
from scc_dags import get_scc_dag, get_ordered_states, get_attractor_states


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

        scc_dag = get_scc_dag(stg)
        scc_indices = get_ordered_states(scc_dag, as_indices=True, DEBUG=True)
        attractor_indices = get_attractor_states(scc_dag, as_indices=True, DEBUG=True)

        transition_matrix = get_transition_matrix(stg, update=update)
        T_inf = nsquare(transition_matrix, 20, DEBUG=True)

        convergence_matrix, _, _ = get_convergence_matrix(T_inf, scc_indices, attractor_indices, DEBUG=True)

        strong_basin = get_strong_basins(convergence_matrix, DEBUG=True)
        expected_strong_basin = np.array([[1],
                                          [0],
                                          [0],
                                          [1],
                                          [1],
                                          [0],
                                          [0],
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
        scc_dag = get_scc_dag(stg)
        scc_indices = get_ordered_states(scc_dag, as_indices=True, DEBUG=True)
        attractor_indices = get_attractor_states(scc_dag, as_indices=True, DEBUG=True)

        transition_matrix = get_transition_matrix(stg, update=update)
        null_group = null_grouping(bnet)
        null_matrix = compress_matrix(transition_matrix, null_group)

        null_inf = nsquare(null_matrix, 20, DEBUG=True)
        null_inf_expanded = expand_matrix(null_inf, null_group)

        convergence_matrix, _, _ = get_convergence_matrix(null_inf_expanded, scc_indices, attractor_indices, DEBUG=True)

        strong_basin = get_strong_basins(convergence_matrix, DEBUG=True)
        expected_strong_basin = np.array([[0],
                                          [0],
                                          [0],
                                          [0],
                                          [0],
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
        scc_dag = get_scc_dag(stg)
        scc_indices = get_ordered_states(scc_dag, as_indices=True, DEBUG=True)
        attractor_indices = get_attractor_states(scc_dag, as_indices=True, DEBUG=True)

        transition_matrix = get_transition_matrix(stg, update=update)
        sd_group = sd_grouping(bnet)
        sd_matrix = compress_matrix(transition_matrix, sd_group)

        sd_inf = nsquare(sd_matrix, 20, DEBUG=True)
        sd_inf_expanded = expand_matrix(sd_inf, sd_group)

        convergence_matrix, _, _ = get_convergence_matrix(sd_inf_expanded, scc_indices, attractor_indices, DEBUG=True)

        strong_basin = get_strong_basins(convergence_matrix, DEBUG=True)
        expected_strong_basin = np.array([[1],
                                          [0],
                                          [0],
                                          [1],
                                          [1],
                                          [0],
                                          [0],
                                          [1],
                                          [1],
                                          [1]])

        self.assertTrue(np.allclose(strong_basin, expected_strong_basin))


class TestGetBasinRatios(unittest.TestCase):
    def test_simple_transition_matrix(self):
        T_inf = np.array([[0, 1], [0, 1]])
        attractor_indices = [[1]]
        basin_ratios = get_basin_ratios(T_inf, attractor_indices)
        expected_basin_ratios = {(1,):1}
        self.assertEqual(basin_ratios, expected_basin_ratios)

    def test_with_attractor_indices(self):
        T_inf = np.array([[0, 1], [0, 1]])
        attractor_indices = [[1]]
        basin_ratios = get_basin_ratios(T_inf, attractor_indices)
        expected_basin_ratios = {(1,):1}
        self.assertEqual(basin_ratios, expected_basin_ratios)

    def test_with_debug(self):
        T_inf = np.array([[0, 1], [0, 1]])
        attractor_indices = [[1]]
        basin_ratios = get_basin_ratios(T_inf, attractor_indices, DEBUG=True)
        expected_basin_ratios = {(1,):1}
        self.assertEqual(basin_ratios, expected_basin_ratios)

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
        T_inf = nsquare(transition_matrix, 20)

        basin_ratios = get_basin_ratios(T_inf, attractor_indices=[[8, 10], [3], [0, 1, 2]])
        expected_basin_ratios = {(8, 10): 0.625, (3,): 0.1, (0, 1, 2): 0.275}
        self.assertTrue(np.allclose(list(basin_ratios.values()), list(expected_basin_ratios.values())))

if __name__ == '__main__':
    unittest.main()