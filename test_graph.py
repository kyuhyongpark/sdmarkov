import unittest

import networkx as nx
import numpy as np
from pyboolnet.external.bnet2primes import bnet_text2primes
from pyboolnet.state_transition_graphs import primes2stg

from graph import get_stg, check_stg, get_markov_chain
from transition_matrix import get_transition_matrix


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


class TestGetMarkovChain(unittest.TestCase):
    def test_empty_compressed_transition_matrix(self):
        compressed_transition_matrix = np.array([])
        group_indices = []
        result = get_markov_chain(compressed_transition_matrix, group_indices)
        self.assertIsInstance(result, nx.DiGraph)
        self.assertEqual(len(result.nodes), 0)
        self.assertEqual(len(result.edges), 0)

    def test_non_square_compressed_transition_matrix_debug_true(self):
        compressed_transition_matrix = np.array([[0.5, 0.5], [0.3, 0.7], [0.1, 0.9]])
        group_indices = [[0, 1], [2]]
        with self.assertRaises(ValueError):
            get_markov_chain(compressed_transition_matrix, group_indices, DEBUG=True)

    def test_compressed_transition_matrix_elements_outside_range_debug_true(self):
        compressed_transition_matrix = np.array([[0.5, 1.5], [0.3, 0.7]])
        group_indices = [[0, 1], [2]]
        with self.assertRaises(ValueError):
            get_markov_chain(compressed_transition_matrix, group_indices, DEBUG=True)

    def test_compressed_transition_matrix_rows_not_summing_to_one_debug_true(self):
        compressed_transition_matrix = np.array([[0.5, 0.4], [0.3, 0.7]])
        group_indices = [[0, 1], [2]]
        with self.assertRaises(ValueError):
            get_markov_chain(compressed_transition_matrix, group_indices, DEBUG=True)

    def test_valid_compressed_transition_matrix_with_group_indices(self):
        compressed_transition_matrix = np.array([[0.5, 0.5], [0.3, 0.7]])
        group_indices = [[0, 1], [], [2, 3]]
        result = get_markov_chain(compressed_transition_matrix, group_indices)
        self.assertIsInstance(result, nx.DiGraph)
        self.assertEqual(len(result.nodes), 2)
        self.assertEqual(len(result.edges), 4)
        self.assertEqual(list(result.nodes), ['G0', 'G2'])
        self.assertEqual(list(result.edges), [('G0', 'G0'), ('G0', 'G2'), ('G2', 'G0'), ('G2', 'G2')])
        self.assertEqual(result.nodes['G0']['indices'], [0, 1])
        self.assertEqual(result.nodes['G0']['states'], ['00', '01'])
        self.assertEqual(result.nodes['G2']['indices'], [2, 3])
        self.assertEqual(result.nodes['G2']['states'], ['10', '11'])
        self.assertEqual(result.edges[('G0', 'G0')]['weight'], 0.5)
        self.assertEqual(result.edges[('G0', 'G2')]['weight'], 0.5)
        self.assertEqual(result.edges[('G2', 'G0')]['weight'], 0.3)
        self.assertEqual(result.edges[('G2', 'G2')]['weight'], 0.7)

    def test_valid_compressed_transition_matrix_with_empty_group_indices(self):
        compressed_transition_matrix = np.array([[0.5, 0.5], [0.3, 0.7]])
        group_indices = []
        with self.assertRaises(ValueError):
            get_markov_chain(compressed_transition_matrix, group_indices, DEBUG=True)

    def test_example(self):
        compressed_transition_matrix = np.array([[0.75 , 0.062, 0.062, 0.062, 0.   , 0.062, 0.   ],
                                                 [0.   , 1.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
                                                 [0.   , 0.   , 1.   , 0.   , 0.   , 0.   , 0.   ],
                                                 [0.   , 0.   , 0.   , 0.875, 0.125, 0.   , 0.   ],
                                                 [0.   , 0.   , 0.   , 0.   , 1.   , 0.   , 0.   ],
                                                 [0.   , 0.   , 0.   , 0.25 , 0.   , 0.625, 0.125],
                                                 [0.   , 0.   , 0.   , 0.   , 0.25 , 0.   , 0.75 ]])
        group_indices = [[4, 5, 6, 7], [], [0, 1, 2], [3], [12, 14], [8, 10], [13, 15], [9, 11]]
        result = get_markov_chain(compressed_transition_matrix, group_indices)
        self.assertIsInstance(result, nx.DiGraph)
        self.assertEqual(sorted(list(result.nodes)), sorted(['G0', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7']))
        self.assertEqual(sorted(list(result.edges)), sorted([('G0', 'G0'), ('G0', 'G2'), ('G0', 'G3'), ('G0', 'G4'), ('G0', 'G6'),
                                                             ('G2', 'G2'),
                                                             ('G3', 'G3'),
                                                             ('G4', 'G4'), ('G4', 'G5'),
                                                             ('G5', 'G5'),
                                                             ('G6', 'G4'), ('G6', 'G6'), ('G6', 'G7'),
                                                             ('G7', 'G5'), ('G7', 'G7')]))


if __name__ == '__main__':
    unittest.main()