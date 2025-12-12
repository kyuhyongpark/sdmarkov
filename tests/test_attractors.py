import unittest

import numpy as np
from pyboolnet.external.bnet2primes import bnet_text2primes
from pyboolnet.state_transition_graphs import primes2stg

from sdmarkov.attractors import attractor_or_transient, get_predicted_attractors
from sdmarkov.transition_matrix import get_transition_matrix
from sdmarkov.matrix_operations import compress_matrix
from sdmarkov.grouping import sd_grouping, null_grouping


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


class TestGetPredictedAttractors(unittest.TestCase):
    def test_non_grouped_transition_matrix(self):
        transition_matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
        attractor_states = get_predicted_attractors(transition_matrix)
        self.assertEqual(attractor_states, [['0', '1']])

    def test_grouped_transition_matrix(self):
        transition_matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
        group_indices = [[0, 1], [2, 3]]
        attractor_states = get_predicted_attractors(transition_matrix, group_indices)
        self.assertEqual(attractor_states, [['00', '01', '10', '11']])

    def test_as_indices_true(self):
        transition_matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
        attractor_states = get_predicted_attractors(transition_matrix, as_indices=True)
        self.assertEqual(attractor_states, [[0, 1]])

    def test_debug_true(self):
        transition_matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
        attractor_states = get_predicted_attractors(transition_matrix, DEBUG=True)
        self.assertEqual(attractor_states, [['0', '1']])
        
    def test_invalid_input_invalid_group_indices(self):
        transition_matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
        group_indices = [[0, 2]]  # invalid index
        with self.assertRaises(ValueError):
            get_predicted_attractors(transition_matrix, group_indices, DEBUG=True)

    def test_example_real(self):
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

        attractor_states = get_predicted_attractors(transition_matrix)

        expected_attractor_states = [["0000", "0001", "0010"], ["0011"], ["1000", "1010"]]
        self.assertEqual(attractor_states, expected_attractor_states)

    def test_example_sd(self):
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

        sd_indices = sd_grouping(bnet, DEBUG=True)
        Tsd = compress_matrix(transition_matrix, sd_indices, DEBUG=True)

        attractor_states = get_predicted_attractors(Tsd, sd_indices, DEBUG=True)

        expected_attractor_states = [["0000", "0001", "0010"], ["0011"], ["1000", "1010"]]
        self.assertEqual(attractor_states, expected_attractor_states)

    def test_example_null(self):
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

        null_indices = null_grouping(bnet, DEBUG=True)
        Tnull = compress_matrix(transition_matrix, null_indices, DEBUG=True)

        attractor_states = get_predicted_attractors(Tnull, null_indices, DEBUG=True)

        expected_attractor_states = [["0011"], ["1000", "1010"]]
        self.assertEqual(attractor_states, expected_attractor_states)


if __name__ == '__main__':
    unittest.main()