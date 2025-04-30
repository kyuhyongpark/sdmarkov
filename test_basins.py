import unittest

import numpy as np
from pyboolnet.external.bnet2primes import bnet_text2primes
from pyboolnet.state_transition_graphs import primes2stg

from basins import get_strong_basins, compare_strong_basins, get_basin_ratios
from transition_matrix import get_transition_matrix
from grouping import sd_grouping, null_grouping
from matrix_operations import nsquare, compress_matrix

class TestGetStrongBasins(unittest.TestCase):
    def test_simple_transition_matrix(self):
        transition_matrix = np.array([[0, 1], [1, 0]])
        attractor_indexes = [[0, 1]]
        strong_basin = get_strong_basins(transition_matrix, attractor_indexes)
        expected_strong_basin = np.array([[0], [0]])
        self.assertTrue(np.allclose(strong_basin, expected_strong_basin))

    def test_no_attractors(self):
        transition_matrix = np.array([[0, 0], [0, 0]])
        attractor_indexes = [[0, 1]]
        with self.assertRaises(ValueError):
            get_strong_basins(transition_matrix, attractor_indexes, DEBUG=True)

    def test_wrong_attractors(self):
        transition_matrix = np.array([[1, 0], [0, 1]])
        attractor_indexes = [[0]]
        with self.assertRaises(ValueError):
            get_strong_basins(transition_matrix, attractor_indexes, DEBUG=True)

    def test_multiple_attractors(self):
        transition_matrix = np.array([[1, 0], [0, 1]])
        attractor_indexes = [[0], [1]]
        strong_basin = get_strong_basins(transition_matrix, attractor_indexes)
        expected_strong_basin = np.array([[0], [1]])
        self.assertTrue(np.allclose(strong_basin, expected_strong_basin))

    def test_with_grouped_transition_matrix(self):
        transition_matrix = np.array([[0, 1], [0, 1]])
        attractor_indexes = [[2]]
        group_indexes = [[0, 1]]
        strong_basins = get_strong_basins(transition_matrix, attractor_indexes, grouped=True, group_indexes=group_indexes)
        expected_strong_basins = np.array([[0], [0], [0]])
        np.testing.assert_array_equal(strong_basins, expected_strong_basins)

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
        attractor_indexes = [[0, 1, 2], [3], [8, 10]]
        strong_basin = get_strong_basins(transition_matrix, attractor_indexes, DEBUG=True)
        expected_strong_basin = np.array([[ 0],
                                          [ 0],
                                          [ 0],
                                          [ 1],
                                          [-1],
                                          [-1],
                                          [-1],
                                          [-1],
                                          [ 2],
                                          [ 2],
                                          [ 2],
                                          [ 2],
                                          [ 2],
                                          [ 2],
                                          [ 2],
                                          [ 2]])

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

        null_group = null_grouping(bnet)

        transition_matrix = get_transition_matrix(stg, update=update)
        null_matrix = compress_matrix(transition_matrix, null_group)
        attractor_indexes = [[0, 1, 2], [3], [8, 10]]
        strong_basin = get_strong_basins(null_matrix, attractor_indexes, grouped=True, group_indexes=null_group, DEBUG=True)
        expected_strong_basin = np.array([[-1],
                                          [-1],
                                          [-1],
                                          [ 1],
                                          [-1],
                                          [-1],
                                          [-1],
                                          [-1],
                                          [ 2],
                                          [-1],
                                          [ 2],
                                          [-1],
                                          [-1],
                                          [-1],
                                          [-1],
                                          [-1]])

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

        sd_group = sd_grouping(bnet)

        transition_matrix = get_transition_matrix(stg, update=update)
        sd_matrix = compress_matrix(transition_matrix, sd_group)
        attractor_indexes = [[0, 1, 2], [3], [8, 10]]
        strong_basin = get_strong_basins(sd_matrix, attractor_indexes, grouped=True, group_indexes=sd_group, DEBUG=True)
        expected_strong_basin = np.array([[ 0],
                                          [ 0],
                                          [ 0],
                                          [ 1],
                                          [-1],
                                          [-1],
                                          [-1],
                                          [-1],
                                          [ 2],
                                          [ 2],
                                          [ 2],
                                          [ 2],
                                          [ 2],
                                          [ 2],
                                          [ 2],
                                          [ 2]])

        self.assertTrue(np.allclose(strong_basin, expected_strong_basin))

class TestCompareStrongBasins(unittest.TestCase):
    def test_same_shape_no_errors(self):
        answer = np.array([[1], [1], [1], [1]])
        guess = np.array([[1], [1], [1], [1]])
        TP, FP, TN, FN = compare_strong_basins(answer, guess)
        self.assertEqual(TP, 4)
        self.assertEqual(FP, 0)
        self.assertEqual(TN, 0)
        self.assertEqual(FN, 0)

    def test_same_shape_with_errors(self):
        answer = np.array([[1], [1], [1], [1]])
        guess = np.array([[1], [1], [1], [-1]])
        TP, FP, TN, FN = compare_strong_basins(answer, guess, DEBUG=True)
        self.assertEqual(TP, 3)
        self.assertEqual(FP, 0)
        self.assertEqual(TN, 0)
        self.assertEqual(FN, 1)

    def test_different_shape(self):
        answer = np.array([[1], [1], [1], [1]])
        guess = np.array([[1], [1], [1]])
        with self.assertRaises(ValueError):
            compare_strong_basins(answer, guess, DEBUG=True)

    def test_false_positives(self):
        answer = np.array([[-1], [-1], [-1], [-1]])
        guess = np.array([[1], [1], [1], [1]])
        with self.assertRaises(ValueError):
            compare_strong_basins(answer, guess, DEBUG=True)


class TestGetBasinRatios(unittest.TestCase):
    def test_simple_transition_matrix(self):
        T_inf = np.array([[0, 1], [0, 1]])
        attractor_indexes = [[1]]
        basin_ratios = get_basin_ratios(T_inf, attractor_indexes)
        expected_basin_ratios = {(1,):1}
        self.assertEqual(basin_ratios, expected_basin_ratios)

    def test_with_attractor_indexes(self):
        T_inf = np.array([[0, 1], [0, 1]])
        attractor_indexes = [[1]]
        basin_ratios = get_basin_ratios(T_inf, attractor_indexes)
        expected_basin_ratios = {(1,):1}
        self.assertEqual(basin_ratios, expected_basin_ratios)

    def test_with_debug(self):
        T_inf = np.array([[0, 1], [0, 1]])
        attractor_indexes = [[1]]
        basin_ratios = get_basin_ratios(T_inf, attractor_indexes, DEBUG=True)
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

        basin_ratios = get_basin_ratios(T_inf, attractor_indexes=[[8, 10], [3], [0, 1, 2]])
        expected_basin_ratios = {(8, 10): 0.625, (3,): 0.1, (0, 1, 2): 0.275}
        self.assertTrue(np.allclose(list(basin_ratios.values()), list(expected_basin_ratios.values())))

if __name__ == '__main__':
    unittest.main()