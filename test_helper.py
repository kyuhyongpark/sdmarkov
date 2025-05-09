import unittest

import networkx as nx
import numpy as np

from helper import states_to_indices, indices_to_states, check_transition_matrix, check_stg


class TestStatesToIndices(unittest.TestCase):
    def test_valid_input(self):
        input_data = [['0100', '0101', '0110', '0111', '1101', '1111'], 
                      ['1001', '1011'], 
                      ['0000', '0001', '0010'], 
                      ['0011'], 
                      ['1100', '1110'], 
                      ['1000', '1010']]
        expected_output = [[4, 5, 6, 7, 13, 15], 
                           [9, 11], 
                           [0, 1, 2], 
                           [3], 
                           [12, 14], 
                           [8, 10]]
        self.assertEqual(states_to_indices(input_data), expected_output)

    def test_empty_input(self):
        input_data = []
        expected_output = []
        self.assertEqual(states_to_indices(input_data), expected_output)

    def test_single_element_input(self):
        input_data = [['0100']]
        expected_output = [[4]]
        self.assertEqual(states_to_indices(input_data), expected_output)

    def test_debug_mode_enabled(self):
        input_data = [['110', '001', '010'], ['111', '000']]
        expected_output = [[6, 1, 2], [7, 0]]
        self.assertEqual(states_to_indices(input_data, DEBUG=True), expected_output)

    def test_invalid_input_different_length(self):
        input_data = [['0100', '0101', '0110', '0111', '1101', '1111'], 
                      ['1001', '10111'], 
                      ['0000', '0001', '0010'], 
                      ['0011'], 
                      ['1100', '1110'], 
                      ['1000', '1010']]
        with self.assertRaises(ValueError):
            states_to_indices(input_data, DEBUG=True)

    def test_invalid_input_duplicates(self):
        input_data = [['0100', '0101', '0110', '0111', '1101', '1111'], 
                      ['1001', '1001'], 
                      ['0000', '0001', '0010'], 
                      ['0011'], 
                      ['1100', '1110'], 
                      ['1000', '1010']]
        with self.assertRaises(ValueError):
            states_to_indices(input_data, DEBUG=True)


class TestIndicesToStates(unittest.TestCase):
    def test_empty_input(self):
        input_data = []
        expected_output = []
        self.assertEqual(indices_to_states(input_data, 4), expected_output)
    
    def test_single_element_input(self):
        input_data = [[4]]
        expected_output = [['0100']]
        self.assertEqual(indices_to_states(input_data, 4), expected_output)

    def test_valid_input(self):
        input_data = [[4, 5, 6, 7, 13, 15], 
                      [9, 11], 
                      [0, 1, 2], 
                      [3], 
                      [12, 14], 
                      [8, 10]]
        expected_output = [['0100', '0101', '0110', '0111', '1101', '1111'], 
                           ['1001', '1011'], 
                           ['0000', '0001', '0010'], 
                           ['0011'], 
                           ['1100', '1110'], 
                           ['1000', '1010']]
        self.assertEqual(indices_to_states(input_data, 4), expected_output)

    def test_debug_mode_enabled(self):
        input_data = [[6, 1, 2], [7, 0]]
        expected_output = [['110', '001', '010'], ['111', '000']]
        self.assertEqual(indices_to_states(input_data, 3, DEBUG=True), expected_output)

    def test_too_small_N(self):
        input_data = [[1, 2, 6], [0, 7]]
        with self.assertRaises(ValueError):
            indices_to_states(input_data, 2, DEBUG=True)

    def test_invalid_input_duplicates(self):
        input_data = [[1, 2, 6], [1, 7]]
        with self.assertRaises(ValueError):
            indices_to_states(input_data, 3, DEBUG=True)


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