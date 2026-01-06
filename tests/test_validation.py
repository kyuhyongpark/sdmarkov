import unittest

import networkx as nx
import numpy as np

from sdmarkov.validation import check_transition_matrix, check_stg


class TestCheckTransitionMatrix(unittest.TestCase):
    def test_square_matrix(self):
        transition_matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
        check_transition_matrix(transition_matrix)

    def test_non_square_matrix(self):
        transition_matrix = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        with self.assertRaises(ValueError):
            check_transition_matrix(transition_matrix)

    def test_elements_outside_range(self):
        transition_matrix = np.array([[0.5, 1.5], [0.5, 0.5]])
        with self.assertRaises(ValueError):
            check_transition_matrix(transition_matrix)

    def test_row_does_not_sum_to_1(self):
        transition_matrix = np.array([[0.5, 0.4], [0.5, 0.5]])
        with self.assertRaises(ValueError):
            check_transition_matrix(transition_matrix)

    def test_compressed_matrix_with_2N_dimensions(self):
        transition_matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
        check_transition_matrix(transition_matrix, compressed=True)

    def test_compressed_matrix_without_2N_dimensions(self):
        transition_matrix = np.array([[0.5, 0.5, 0], [0.5, 0.5, 0], [0.5, 0.5, 0]])
        check_transition_matrix(transition_matrix, compressed=True)

    def test_non_compressed_matrix_without_2N_dimensions(self):
        transition_matrix = np.array([[0.5, 0.5, 0], [0.5, 0.5, 0], [0.5, 0.5, 0]])
        with self.assertRaises(ValueError):
            check_transition_matrix(transition_matrix)


class TestCheckSTG(unittest.TestCase):
    def test_valid_stg(self):
        # Create a valid state transition graph
        stg = nx.DiGraph()
        stg.add_nodes_from(['00', '01', '10', '11'])
        stg.add_edge('10', '11')
        stg.add_edge('01', '00')
        check_stg(stg)  # Should not raise an error

    def test_invalid_stg_N_leq_0(self):
        # Create an invalid state transition graph with N <= 0
        stg = nx.DiGraph()
        stg.add_node('')
        with self.assertRaises(ValueError):
            check_stg(stg)

    def test_invalid_stg_non_string_node(self):
        # Create an invalid state transition graph with non-string node
        stg = nx.DiGraph()
        stg.add_node(10)
        with self.assertRaises(ValueError):
            check_stg(stg)

    def test_invalid_stg_node_not_0s_and_1s(self):
        # Create an invalid state transition graph with node not containing only 0s and 1s
        stg = nx.DiGraph()
        stg.add_node('12')
        with self.assertRaises(ValueError):
            check_stg(stg)

    def test_invalid_stg_nodes_different_lengths(self):
        # Create an invalid state transition graph with nodes of different lengths
        stg = nx.DiGraph()
        stg.add_nodes_from(['0', '00'])
        with self.assertRaises(ValueError):
            check_stg(stg)

    def test_invalid_stg_num_nodes_not_2_N(self):
        # Create an invalid state transition graph with number of nodes not equal to 2^N
        stg = nx.DiGraph()
        stg.add_nodes_from(['00', '01', '10'])
        with self.assertRaises(ValueError):
            check_stg(stg)

    def test_invalid_stg_outgoing_transitions_greater_than_N(self):
        # Create an invalid state transition graph with outgoing transitions greater than N
        stg = nx.DiGraph()
        stg.add_nodes_from(['00', '01', '10', '11'])
        stg.add_edge('00', '00')
        stg.add_edge('00', '01')
        stg.add_edge('00', '10')

        with self.assertRaises(ValueError):
            check_stg(stg)


if __name__ == '__main__':
    unittest.main()