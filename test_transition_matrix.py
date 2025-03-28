import unittest

import numpy as np
import networkx as nx
from pyboolnet.state_transition_graphs import primes2stg
from pyboolnet.external.bnet2primes import bnet_text2primes

from transition_matrix import check_stg, check_transition_matrix
from transition_matrix import get_transition_matrix, get_hamming_distance_matrix, get_bitflip_matrix
from transition_matrix import get_stg
from transition_matrix import get_markov_chain


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


class TestGetTransitionMatrix(unittest.TestCase):
    def test_empty_stg(self):
        # Test case: Empty stg
        stg = nx.DiGraph()
        self.assertIsNone(get_transition_matrix(stg))

    def test_invalid_graph_with_debug(self):
        # Test case: Invalid stg with a different lengths
        stg = nx.DiGraph()
        stg.add_nodes_from(['0', '01'])
        with self.assertRaises(ValueError):
            get_transition_matrix(stg, DEBUG=True)

        # Test case: Invalid stg with number of states != 2**N
        stg = nx.DiGraph()
        stg.add_nodes_from(['00', '01', '10'])
        with self.assertRaises(ValueError):
            get_transition_matrix(stg, DEBUG=True)

    def test_graph_with_single_node(self):
        # Test case: Graph with a single node, and hence two states
        stg = nx.DiGraph()
        stg.add_nodes_from(['0', '1'])
        expected = np.array([[1, 0], [0, 1]])
        result = get_transition_matrix(stg)
        self.assertTrue(np.allclose(result, expected))
        self.assertTrue(np.allclose(np.sum(result, axis=1), np.ones(2)))

    def test_graph_with_multiple_nodes(self):
        # Test case: Graph with multiple nodes and edges
        stg = nx.DiGraph()
        stg.add_nodes_from(['00', '01', '10', '11'])
        stg.add_edge('01', '00')
        stg.add_edge('10', '00')
        expected = np.array([
            [1, 0, 0, 0],
            [1/2, 1/2, 0, 0],
            [1/2, 0, 1/2, 0],
            [0, 0, 0, 1]
        ])
        result = get_transition_matrix(stg)
        self.assertTrue(np.allclose(result, expected))
        self.assertTrue(np.allclose(np.sum(result, axis=1), np.ones(4)))

    def test_graph_with_self_loops(self):
        # Test case: Graph with self-loops
        stg = nx.DiGraph()
        stg.add_nodes_from(['0', '1'])
        stg.add_edge('0', '0')
        stg.add_edge('1', '0')
        expected = np.array([
            [1, 0],
            [1, 0]])
        result = get_transition_matrix(stg)
        self.assertTrue(np.allclose(result, expected))
        self.assertTrue(np.allclose(np.sum(result, axis=1), np.ones(2)))

    def test_with_bnet_example(self):
        # Test case: stg from bnet example
        bnet = """
               targets, factors
               A, A | B & C
               B, B & !C
               C, B & !C | !C & !D | !B & C & D
               D, !A & !B & !C & !D | !A & C & D
               """
        primes = bnet_text2primes(bnet)
        update = "asynchronous"
        stg = primes2stg(primes, update)

        expected = np.array(
            [[1/2, 1/4, 1/4, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  ],
             [1/4, 3/4, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  ],
             [1/4, 0,   3/4, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  ],
             [0,   0,   0,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  ],
             [0,   0,   0,   0,   3/4, 0,   1/4, 0,   0,   0,   0,   0,   0,   0,   0,   0,  ],
             [0,   0,   0,   0,   1/4, 1/2, 0,   1/4, 0,   0,   0,   0,   0,   0,   0,   0,  ],
             [0,   0,   1/4, 0,   1/4, 0,   1/4, 0,   0,   0,   0,   0,   0,   0,   1/4, 0,  ],
             [0,   0,   0,   1/4, 0,   1/4, 0,   1/4, 0,   0,   0,   0,   0,   0,   0,   1/4 ],
             [0,   0,   0,   0,   0,   0,   0,   0,   3/4, 0,   1/4, 0,   0,   0,   0,   0,  ],
             [0,   0,   0,   0,   0,   0,   0,   0,   1/4, 3/4, 0,   0,   0,   0,   0,   0,  ],
             [0,   0,   0,   0,   0,   0,   0,   0,   1/4, 0,   3/4, 0,   0,   0,   0,   0,  ],
             [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1/4, 3/4, 0,   0,   0,   0,  ],
             [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   3/4, 0,   1/4, 0,  ],
             [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1/4, 1/2, 0,   1/4 ],
             [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1/4, 0,   1/4, 0,   1/2, 0,  ],
             [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1/4, 0,   1/4, 1/4, 1/4 ]])

        result = get_transition_matrix(stg)
        self.assertTrue(np.allclose(result, expected))
        self.assertTrue(np.allclose(np.sum(result, axis=1), np.ones(16)))


class TestGetHammingDistanceMatrix(unittest.TestCase):
    def test_empty_stg(self):
        stg = nx.DiGraph()
        result = get_hamming_distance_matrix(stg=stg)
        self.assertIsNone(result)

    def test_multiple_states_stg(self):
        stg = nx.DiGraph()
        stg.add_nodes_from(['00', '01', '10', '11'])
        result = get_hamming_distance_matrix(stg=stg)
        expected = np.array([[0, 1, 1, 2], [1, 0, 2, 1], [1, 2, 0, 1], [2, 1, 1, 0]])
        self.assertTrue(np.allclose(result, expected))

    def test_with_bnet_example(self):
        # Test case: stg from bnet example
        bnet = """
               targets, factors
               A, A | B & C
               B, B & !C
               C, B & !C | !C & !D | !B & C & D
               D, !A & !B & !C & !D | !A & C & D
               """    
        primes = bnet_text2primes(bnet)
        update = "asynchronous"
        stg = primes2stg(primes, update)
        expected = np.array(
            [[0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,],
             [1, 0, 2, 1, 2, 1, 3, 2, 2, 1, 3, 2, 3, 2, 4, 3,],
             [1, 2, 0, 1, 2, 3, 1, 2, 2, 3, 1, 2, 3, 4, 2, 3,],
             [2, 1, 1, 0, 3, 2, 2, 1, 3, 2, 2, 1, 4, 3, 3, 2,],
             [1, 2, 2, 3, 0, 1, 1, 2, 2, 3, 3, 4, 1, 2, 2, 3,],
             [2, 1, 3, 2, 1, 0, 2, 1, 3, 2, 4, 3, 2, 1, 3, 2,],
             [2, 3, 1, 2, 1, 2, 0, 1, 3, 4, 2, 3, 2, 3, 1, 2,],
             [3, 2, 2, 1, 2, 1, 1, 0, 4, 3, 3, 2, 3, 2, 2, 1,],
             [1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3,],
             [2, 1, 3, 2, 3, 2, 4, 3, 1, 0, 2, 1, 2, 1, 3, 2,],
             [2, 3, 1, 2, 3, 4, 2, 3, 1, 2, 0, 1, 2, 3, 1, 2,],
             [3, 2, 2, 1, 4, 3, 3, 2, 2, 1, 1, 0, 3, 2, 2, 1,],
             [2, 3, 3, 4, 1, 2, 2, 3, 1, 2, 2, 3, 0, 1, 1, 2,],
             [3, 2, 4, 3, 2, 1, 3, 2, 2, 1, 3, 2, 1, 0, 2, 1,],
             [3, 4, 2, 3, 2, 3, 1, 2, 2, 3, 1, 2, 1, 2, 0, 1,],
             [4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,]])
        
        result = get_hamming_distance_matrix(stg=stg)
        self.assertTrue(np.allclose(result, expected))

    def test_N_specified(self):
        N = 3
        expected_matrix = np.array([[0, 1, 1, 2, 1, 2, 2, 3],
                                    [1, 0, 2, 1, 2, 1, 3, 2],
                                    [1, 2, 0, 1, 2, 3, 1, 2],
                                    [2, 1, 1, 0, 3, 2, 2, 1],
                                    [1, 2, 2, 3, 0, 1, 1, 2],
                                    [2, 1, 3, 2, 1, 0, 2, 1],
                                    [2, 3, 1, 2, 1, 2, 0, 1],
                                    [3, 2, 2, 1, 2, 1, 1, 0]])
        result_matrix = get_hamming_distance_matrix(N=N)
        self.assertTrue(np.array_equal(result_matrix, expected_matrix))

class TestGetBitflipMatrix(unittest.TestCase):
    def test_empty_Hamming_distance_matrix(self):
        hd = np.array([])
        size = 1
        self.assertIsNone(get_bitflip_matrix(hd, size))    

    def test_size_is_greater_than_max_hd(self):
        hd = np.array([[0, 1], [1, 0]])
        size = 2
        with self.assertRaises(ValueError):
            get_bitflip_matrix(hd, size)

    def test_size_is_less_than_1(self):
        hd = np.array([[0, 1], [1, 0]])
        size = 0
        with self.assertRaises(ValueError):
            get_bitflip_matrix(hd, size)

    def test_hd_is_not_square_matrix(self):
        hd = np.array([[0, 1, 2], [1, 0, 1]])
        size = 1
        with self.assertRaises(ValueError):
            get_bitflip_matrix(hd, size, DEBUG=True)

    def test_Hamming_distance_matrix_with_non_integer_values(self):
        hd = np.array([[0, 1.5], [1, 0]])
        size = 1
        with self.assertRaises(ValueError):
            get_bitflip_matrix(hd, size, DEBUG=True)

    def test_Hamming_distance_matrix_with_values_outside_range(self):
        hd = np.array([[0, 2], [1, 0]])
        size = 1
        with self.assertRaises(ValueError):
            get_bitflip_matrix(hd, size, DEBUG=True)

    def test_hd_not_have_all_values(self):
        hd = np.array([[0, 1, 1, 2],
                       [1, 0, 1, 1],
                       [1, 2, 0, 1],
                       [2, 1, 1, 0]])
        size = 1
        with self.assertRaises(ValueError):
            get_bitflip_matrix(hd, size, DEBUG=True)

    def test_size_is_1(self):
        hd = np.array([[0, 1, 1, 2],
                       [1, 0, 2, 1],
                       [1, 2, 0, 1],
                       [2, 1, 1, 0]])
        size = 1
        expected = np.array(
            [[0, 1/2, 1/2, 0],
             [1/2, 0, 0, 1/2],
             [1/2, 0, 0, 1/2],
             [0, 1/2, 1/2, 0]])
        self.assertTrue(np.allclose(get_bitflip_matrix(hd, size), expected))

    def test_size_is_max(self):
        hd = np.array([[0, 1, 1, 2],
                       [1, 0, 2, 1],
                       [1, 2, 0, 1],
                       [2, 1, 1, 0]])
        size = 2
        expected = np.array(
            [[0, 0, 0, 1],
             [0, 0, 1, 0],
             [0, 1, 0, 0],
             [1, 0, 0, 0]])
        self.assertTrue(np.allclose(get_bitflip_matrix(hd, size), expected))


class TestGetSTG(unittest.TestCase):
    def test_empty_transition_matrix(self):
        transition_matrix = np.array([])
        stg = get_stg(transition_matrix)
        self.assertIsInstance(stg, nx.DiGraph)
        self.assertEqual(len(stg.nodes), 0)

    def test_non_square_transition_matrix(self):
        transition_matrix = np.array([[0, 1], [1, 0], [0, 1]])
        with self.assertRaises(ValueError):
            get_stg(transition_matrix, DEBUG=True)

    def test_transition_matrix_with_elements_outside_01(self):
        transition_matrix = np.array([[0, 2], [1, 0]])
        with self.assertRaises(ValueError):
            get_stg(transition_matrix, DEBUG=True)

    def test_transition_matrix_with_rows_not_summing_to_1(self):
        transition_matrix = np.array([[0, 2], [1, 0]])
        with self.assertRaises(ValueError):
            get_stg(transition_matrix, DEBUG=True)

    def test_valid_transition_matrix_with_multiple_states(self):
        transition_matrix = np.array([[0.5, 0.5], [0.3, 0.7]])
        stg = get_stg(transition_matrix)
        self.assertIsInstance(stg, nx.DiGraph)
        self.assertEqual(len(stg.nodes), 2)
        self.assertEqual(len(stg.edges), 2)

    def test_valid_transition_matrix_with_single_state(self):
        transition_matrix = np.array([[1]])
        stg = get_stg(transition_matrix)
        self.assertIsInstance(stg, nx.DiGraph)
        self.assertEqual(len(stg.nodes), 1)
        self.assertEqual(len(stg.edges), 1)

    def test_tm2stg2tm(self):
        transition_matrix =np.array(
            [[1/2, 1/4, 1/4, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  ],
             [1/4, 3/4, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  ],
             [1/4, 0,   3/4, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  ],
             [0,   0,   0,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  ],
             [0,   0,   0,   0,   3/4, 0,   1/4, 0,   0,   0,   0,   0,   0,   0,   0,   0,  ],
             [0,   0,   0,   0,   1/4, 1/2, 0,   1/4, 0,   0,   0,   0,   0,   0,   0,   0,  ],
             [0,   0,   1/4, 0,   1/4, 0,   1/4, 0,   0,   0,   0,   0,   0,   0,   1/4, 0,  ],
             [0,   0,   0,   1/4, 0,   1/4, 0,   1/4, 0,   0,   0,   0,   0,   0,   0,   1/4 ],
             [0,   0,   0,   0,   0,   0,   0,   0,   3/4, 0,   1/4, 0,   0,   0,   0,   0,  ],
             [0,   0,   0,   0,   0,   0,   0,   0,   1/4, 3/4, 0,   0,   0,   0,   0,   0,  ],
             [0,   0,   0,   0,   0,   0,   0,   0,   1/4, 0,   3/4, 0,   0,   0,   0,   0,  ],
             [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1/4, 3/4, 0,   0,   0,   0,  ],
             [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   3/4, 0,   1/4, 0,  ],
             [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1/4, 1/2, 0,   1/4 ],
             [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1/4, 0,   1/4, 0,   1/2, 0,  ],
             [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1/4, 0,   1/4, 1/4, 1/4 ]])

        stg = get_stg(transition_matrix, DEBUG=True)
        transition_matrix2 = get_transition_matrix(stg, DEBUG=True)
        self.assertTrue(np.array_equal(transition_matrix, transition_matrix2))

    def test_bnet2stg2tm2stg2tm(self):
        bnet = """
               targets, factors
               A, A | B & C
               B, B & !C
               C, B & !C | !C & !D | !B & C & D
               D, !A & !B & !C & !D | !A & C & D
               """
        primes = bnet_text2primes(bnet)
        update = "asynchronous"
        stg = primes2stg(primes, update)

        transition_matrix = get_transition_matrix(stg, DEBUG=True)
        stg2 = get_stg(transition_matrix, DEBUG=True)
        transition_matrix2 = get_transition_matrix(stg2, DEBUG=True)

        self.assertTrue(np.array_equal(transition_matrix, transition_matrix2))
        # check if all nodes in stg are in stg2
        for node in stg.nodes:
            self.assertTrue(node in stg2.nodes)
        # check if all nodes in stg2 are in stg
        for node in stg2.nodes:
            self.assertTrue(node in stg.nodes)

        # check if all edges in stg are in stg2
        for edge in stg.edges:
            self.assertTrue(edge in stg2.edges)
        # check if all edges in stg2 are in stg
        for edge in stg2.edges:
            self.assertTrue(edge in stg.edges)


class TestGetMarkovChain(unittest.TestCase):
    def test_empty_compressed_transition_matrix(self):
        compressed_transition_matrix = np.array([])
        group_indexes = []
        result = get_markov_chain(compressed_transition_matrix, group_indexes)
        self.assertIsInstance(result, nx.DiGraph)
        self.assertEqual(len(result.nodes), 0)
        self.assertEqual(len(result.edges), 0)

    def test_non_square_compressed_transition_matrix_debug_true(self):
        compressed_transition_matrix = np.array([[0.5, 0.5], [0.3, 0.7], [0.1, 0.9]])
        group_indexes = [[0, 1], [2]]
        with self.assertRaises(ValueError):
            get_markov_chain(compressed_transition_matrix, group_indexes, DEBUG=True)

    def test_compressed_transition_matrix_elements_outside_range_debug_true(self):
        compressed_transition_matrix = np.array([[0.5, 1.5], [0.3, 0.7]])
        group_indexes = [[0, 1], [2]]
        with self.assertRaises(ValueError):
            get_markov_chain(compressed_transition_matrix, group_indexes, DEBUG=True)

    def test_compressed_transition_matrix_rows_not_summing_to_one_debug_true(self):
        compressed_transition_matrix = np.array([[0.5, 0.4], [0.3, 0.7]])
        group_indexes = [[0, 1], [2]]
        with self.assertRaises(ValueError):
            get_markov_chain(compressed_transition_matrix, group_indexes, DEBUG=True)

    def test_valid_compressed_transition_matrix_with_group_indexes(self):
        compressed_transition_matrix = np.array([[0.5, 0.5], [0.3, 0.7]])
        group_indexes = [[0, 1], [2]]
        result = get_markov_chain(compressed_transition_matrix, group_indexes)
        self.assertIsInstance(result, nx.DiGraph)
        self.assertEqual(len(result.nodes), 2)
        self.assertEqual(len(result.edges), 4)

    def test_valid_compressed_transition_matrix_with_empty_group_indexes(self):
        compressed_transition_matrix = np.array([[0.5, 0.5], [0.3, 0.7]])
        group_indexes = []
        with self.assertRaises(ValueError):
            get_markov_chain(compressed_transition_matrix, group_indexes, DEBUG=True)

    def test_example(self):
        compressed_transition_matrix = np.array([[0.75 , 0.062, 0.062, 0.062, 0.   , 0.062, 0.   ],
                                                 [0.   , 1.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
                                                 [0.   , 0.   , 1.   , 0.   , 0.   , 0.   , 0.   ],
                                                 [0.   , 0.   , 0.   , 0.875, 0.125, 0.   , 0.   ],
                                                 [0.   , 0.   , 0.   , 0.   , 1.   , 0.   , 0.   ],
                                                 [0.   , 0.   , 0.   , 0.25 , 0.   , 0.625, 0.125],
                                                 [0.   , 0.   , 0.   , 0.   , 0.25 , 0.   , 0.75 ]])
        group_indexes = [[4, 5, 6, 7], [], [0, 1, 2], [3], [12, 14], [8, 10], [13, 15], [9, 11]]
        result = get_markov_chain(compressed_transition_matrix, group_indexes)
        self.assertIsInstance(result, nx.DiGraph)
        self.assertEqual(sorted(list(result.nodes)), sorted(['0', '2', '3', '4', '5', '6', '7']))
        self.assertEqual(sorted(list(result.edges)), sorted([('0', '0'), ('0', '2'), ('0', '3'), ('0', '4'), ('0', '6'), ('2', '2'), ('3', '3'), ('4', '4'), ('4', '5'), ('6', '4'), ('6', '6'), ('6', '7'), ('5', '5'), ('7', '5'), ('7', '7')]))

if __name__ == '__main__':
    unittest.main()