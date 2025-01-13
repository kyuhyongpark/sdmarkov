import unittest
import numpy as np
import networkx as nx

from pyboolnet.state_transition_graphs import primes2stg
from pyboolnet.external.bnet2primes import bnet_text2primes

from transition_matrix import check_stg
from transition_matrix import get_transition_matrix
from transition_matrix import get_hamming_distance_matrix
from transition_matrix import get_bitflip_matrix


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
        result = get_hamming_distance_matrix(stg)
        self.assertIsNone(result)

    def test_multiple_states_stg(self):
        stg = nx.DiGraph()
        stg.add_nodes_from(['00', '01', '10', '11'])
        result = get_hamming_distance_matrix(stg)
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
        
        result = get_hamming_distance_matrix(stg)
        self.assertTrue(np.allclose(result, expected))


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




if __name__ == '__main__':
    unittest.main()