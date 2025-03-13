import unittest
import numpy as np

from pyboolnet.external.bnet2primes import bnet_text2primes
from pyboolnet.state_transition_graphs import primes2stg

from transition_matrix import get_transition_matrix
from decisions import get_decision_matrix
from decisions import expand_decision_matrix
from decisions import compare_decision_matrices

class TestGetDecisionMatrix(unittest.TestCase):

    def test_no_decision(self):
        transition_matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
        decision_matrix = get_decision_matrix(transition_matrix)
        expected = np.array([[-1, -1], [-1, -1]])
        self.assertTrue(np.allclose(decision_matrix, expected))

    def test_oscillation(self):
        transition_matrix = np.array([[0, 1], [1, 0]])
        decision_matrix = get_decision_matrix(transition_matrix, DEBUG=True)
        expected = np.array([[0, -1], [-1, 0]])
        self.assertTrue(np.allclose(decision_matrix, expected))

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
        decision_matrix = get_decision_matrix(transition_matrix)
        expected = np.array([[-1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                             [-1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                             [-1,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                             [ 0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                             [ 0,  0,  0,  0, -1,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                             [ 0,  0,  0,  0,  1, -1,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0],
                             [ 0,  0,  1,  0, -1,  0, -1,  0,  0,  0,  0,  0,  0,  0,  1,  0],
                             [ 0,  0,  0,  1,  0, -1,  0, -1,  0,  0,  0,  0,  0,  0,  0,  1],
                             [ 0,  0,  0,  0,  0,  0,  0,  0, -1,  0, -1,  0,  0,  0,  0,  0],
                             [ 0,  0,  0,  0,  0,  0,  0,  0, -1, -1,  0,  0,  0,  0,  0,  0],
                             [ 0,  0,  0,  0,  0,  0,  0,  0, -1,  0, -1,  0,  0,  0,  0,  0],
                             [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1,  0,  0,  0,  0],
                             [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0, -1,  0],
                             [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1,  0, -1],
                             [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0, -1,  0, -1,  0],
                             [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0, -1, -1, -1]])
        self.assertTrue(np.allclose(decision_matrix, expected))


class TestExpandDecisionMatrix(unittest.TestCase):
    def test_square_matrix_mutually_exclusive_groups(self):
        matrix = np.array([[-1, 1], [0, -1]])
        index_groups = [[0, 1], [2]]
        expected_result = np.array([[-1, -1, 1], [-1, -1, 1], [0, 0, -1]])
        self.assertTrue(np.allclose(expand_decision_matrix(matrix, index_groups), expected_result))

    def test_non_square_matrix_mutually_exclusive_groups(self):
        matrix = np.array([[-1, 1, 0], [0, -1, 0]])
        index_groups = [[0, 1], [2]]
        with self.assertRaises(ValueError):
            expand_decision_matrix(matrix, index_groups)

    def test_square_matrix_non_mutually_exclusive_groups(self):
        matrix = np.array([[-1, 1], [0, -1]])
        index_groups = [[0, 1], [0, 2]]
        with self.assertRaises(ValueError):
            expand_decision_matrix(matrix, index_groups, DEBUG=True)

    def test_matrix_size_mismatch(self):
        matrix = np.array([[-1, 1], [0, -1]])
        index_groups = [[0, 1], [2], [3]]
        with self.assertRaises(ValueError):
            expand_decision_matrix(matrix, index_groups, DEBUG=True)


class TestCompareDecisionMatrices(unittest.TestCase):

    def test_same_shape(self):
        answer = np.array([[1, -1], [-1, 1]])
        guess = np.array([[1, -1], [-1, 1]])
        TP, FP, TN, FN = compare_decision_matrices(answer, guess)
        self.assertEqual(TP, 2)
        self.assertEqual(FP, 0)
        self.assertEqual(TN, 2)
        self.assertEqual(FN, 0)

    def test_different_shapes(self):
        answer = np.array([[1, -1], [-1, 1]])
        guess = np.array([[1, -1], [-1, 1], [1, -1]])
        with self.assertRaises(ValueError):
            compare_decision_matrices(answer, guess, DEBUG=True)

    def test_true_positives_false_positives_true_negatives_false_negatives(self):
        answer = np.array([[1, -1], [-1, 1]])
        guess = np.array([[1, 1], [-1, -1]])
        TP, FP, TN, FN = compare_decision_matrices(answer, guess)
        self.assertEqual(TP, 1)
        self.assertEqual(FP, 1)
        self.assertEqual(TN, 1)
        self.assertEqual(FN, 1)

    def test_missing_transitions(self):
        answer = np.array([[1, -1], [-1, 1]])
        guess = np.array([[1, 0], [-1, 1]])
        with self.assertRaises(ValueError):
            compare_decision_matrices(answer, guess, DEBUG=True)

    def test_no_missing_transitions(self):
        answer = np.array([[1, -1], [-1, 1]])
        guess = np.array([[1, -1], [-1, 1]])
        TP, FP, TN, FN = compare_decision_matrices(answer, guess, DEBUG=True)
        self.assertEqual(TP, 2)
        self.assertEqual(FP, 0)
        self.assertEqual(TN, 2)
        self.assertEqual(FN, 0)


if __name__ == '__main__':
    unittest.main()