import unittest

import numpy as np
from pyboolnet.external.bnet2primes import bnet_text2primes
from pyboolnet.state_transition_graphs import primes2stg

from transition_matrix import get_stg, get_transition_matrix
from scc_dags import get_scc_dag
from basins import get_strong_basins, expand_strong_basin_matrix, compare_strong_basins


class TestGetStrongBasins(unittest.TestCase):
    def test_simple_transition_matrix(self):
        transition_matrix = np.array([[0, 1], [1, 0]])
        strong_basin = get_strong_basins(transition_matrix)
        expected_strong_basin = np.array([[0], [0]])
        self.assertTrue(np.allclose(strong_basin, expected_strong_basin))

    def test_with_attractor_indexes(self):
        transition_matrix = np.array([[0, 1], [1, 0]])
        attractor_indexes = [[0, 1]]
        strong_basin = get_strong_basins(transition_matrix, attractor_indexes=attractor_indexes)
        expected_strong_basin = np.array([[0], [0]])
        self.assertTrue(np.allclose(strong_basin, expected_strong_basin))

    def test_with_scc_dag(self):
        transition_matrix = np.array([[0, 1], [1, 0]])
        scc_dag = get_scc_dag(get_stg(transition_matrix))
        strong_basin = get_strong_basins(transition_matrix, scc_dag=scc_dag)
        expected_strong_basin = np.array([[0], [0]])
        self.assertTrue(np.allclose(strong_basin, expected_strong_basin))

    def test_with_state_transition_graph(self):
        transition_matrix = np.array([[0, 1], [1, 0]])
        stg = get_stg(transition_matrix)
        strong_basin = get_strong_basins(transition_matrix, stg=stg)
        expected_strong_basin = np.array([[0], [0]])
        self.assertTrue(np.allclose(strong_basin, expected_strong_basin))

    def test_with_all_optional_parameters(self):
        transition_matrix = np.array([[0, 1], [1, 0]])
        attractor_indexes = [[0, 1]]
        scc_dag = get_scc_dag(get_stg(transition_matrix))
        stg = get_stg(transition_matrix)
        strong_basin = get_strong_basins(transition_matrix, attractor_indexes=attractor_indexes, scc_dag=scc_dag, stg=stg)
        expected_strong_basin = np.array([[0], [0]])
        self.assertTrue(np.allclose(strong_basin, expected_strong_basin))

    def test_no_attractors(self):
        transition_matrix = np.array([[0, 0], [0, 0]])
        with self.assertRaises(ValueError):
            get_strong_basins(transition_matrix, DEBUG=True)

    def test_multiple_attractors(self):
        transition_matrix = np.array([[1, 0], [0, 1]])
        strong_basin = get_strong_basins(transition_matrix)
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
        strong_basin = get_strong_basins(transition_matrix)
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


class TestExpandStrongBasinMatrix(unittest.TestCase):
    def test_mutually_exclusive_index_groups(self):
        strong_basin = np.array([[1], [2]])
        index_groups = [[0, 1], [2]]
        expanded_strong_basin = expand_strong_basin_matrix(strong_basin, index_groups)
        expected_output = np.array([[1], [1], [2]])
        self.assertTrue(np.array_equal(expanded_strong_basin, expected_output))

    def test_non_mutually_exclusive_index_groups(self):
        strong_basin = np.array([[1], [2], [3]])
        index_groups = [[0, 1], [1, 2]]
        with self.assertRaises(ValueError):
            expand_strong_basin_matrix(strong_basin, index_groups, DEBUG=True)

    def test_non_empty_index_groups_greater_than_matrix_size(self):
        strong_basin = np.array([[1], [2]])
        index_groups = [[0, 1], [2], [3]]
        with self.assertRaises(ValueError):
            expand_strong_basin_matrix(strong_basin, index_groups, DEBUG=True)

    def test_empty_index_groups(self):
        strong_basin = np.array([[1], [2], [3]])
        index_groups = [[], []]
        expanded_strong_basin = expand_strong_basin_matrix(strong_basin, index_groups)
        expected_output = np.array([[1], [2], [3]])
        self.assertTrue(np.array_equal(expanded_strong_basin, expected_output))

    def test_single_index_group(self):
        strong_basin = np.array([[1], [2], [3]])
        index_groups = [[0, 1, 2]]
        expanded_strong_basin = expand_strong_basin_matrix(strong_basin, index_groups)
        expected_output = np.array([[1], [1], [1], [2], [3]])
        self.assertTrue(np.array_equal(expanded_strong_basin, expected_output))

    def test_multiple_index_groups(self):
        strong_basin = np.array([[1], [2], [3], [4]])
        index_groups = [[0, 3], [1, 2]]
        expanded_strong_basin = expand_strong_basin_matrix(strong_basin, index_groups)
        expected_output = np.array([[1], [2], [2], [1], [3], [4]])
        self.assertTrue(np.array_equal(expanded_strong_basin, expected_output))

    def test_example(self):
        strong_basin = np.array([[-1], [0], [1], [2], [2], [2], [2]])
        index_groups = [[4, 5, 6, 7], [], [0, 1, 2], [3], [12, 14], [8, 10], [13, 15], [9, 11]]
        expanded_strong_basin = expand_strong_basin_matrix(strong_basin, index_groups)
        expected_output = np.array([[ 0.],
                                    [ 0.],
                                    [ 0.],
                                    [ 1.],
                                    [-1.],
                                    [-1.],
                                    [-1.],
                                    [-1.],
                                    [ 2.],
                                    [ 2.],
                                    [ 2.],
                                    [ 2.],
                                    [ 2.],
                                    [ 2.],
                                    [ 2.],
                                    [ 2.]])

        self.assertTrue(np.array_equal(expanded_strong_basin, expected_output))


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


if __name__ == '__main__':
    unittest.main()