import unittest

import networkx as nx
import numpy as np

from paths import get_all_paths, get_markov_chain_path_probs, get_stg_path_probs, solve_matrix_equation
from paths import compare_path_reachability, compare_path_rmsd
from transition_matrix import get_markov_chain

class TestGetAllPaths(unittest.TestCase):
    def test_empty_graph(self):
        graph = nx.DiGraph()
        self.assertEqual(get_all_paths(graph), [])

    def test_graph_with_no_edges(self):
        graph = nx.DiGraph()
        graph.add_nodes_from([1, 2, 3])
        self.assertEqual(get_all_paths(graph), [])

    def test_graph_with_single_edge(self):
        graph = nx.DiGraph()
        graph.add_edge(1, 2)
        expected_paths = [(1, 2)]
        self.assertEqual(get_all_paths(graph), expected_paths)

    def test_graph_with_multiple_edges_and_nodes(self):
        graph = nx.DiGraph()
        graph.add_edges_from([(1, 2), (2, 3), (1, 3)])
        expected_paths = [(1, 2), (1, 3), (2, 3), (1, 2, 3)]
        self.assertEqual(get_all_paths(graph), expected_paths)

    def test_graph_with_self_loops(self):
        graph = nx.DiGraph()
        graph.add_edges_from([(1, 1), (2, 2)])
        expected_paths = []
        self.assertEqual(get_all_paths(graph), expected_paths)

    def test_graph_with_multiple_paths_between_two_nodes(self):
        graph = nx.DiGraph()
        graph.add_edges_from([(1, 2), (1, 3), (3, 2)])
        expected_paths = [(1, 2), (1, 3), (3, 2), (1, 3, 2)]
        self.assertEqual(get_all_paths(graph), expected_paths)

    def test_debug_parameter_set_to_true(self):
        graph = nx.DiGraph()
        graph.add_edge(1, 2)
        expected_paths = [(1, 2)]
        self.assertEqual(get_all_paths(graph, DEBUG=True), expected_paths)

    def test_debug_parameter_set_to_false(self):
        graph = nx.DiGraph()
        graph.add_edge(1, 2)
        expected_paths = [(1, 2)]
        self.assertEqual(get_all_paths(graph, DEBUG=False), expected_paths)

    def test_example(self):
        compressed_transition_matrix = np.array([[0.75 , 0.0625, 0.0625, 0.0625, 0.   , 0.0625, 0.   ],
                                                 [0.   , 1.    , 0.    , 0.    , 0.   , 0.    , 0.   ],
                                                 [0.   , 0.    , 1.    , 0.    , 0.   , 0.    , 0.   ],
                                                 [0.   , 0.    , 0.    , 0.875 , 0.125, 0.    , 0.   ],
                                                 [0.   , 0.    , 0.    , 0.    , 1.   , 0.    , 0.   ],
                                                 [0.   , 0.    , 0.    , 0.25  , 0.   , 0.625 , 0.125],
                                                 [0.   , 0.    , 0.    , 0.    , 0.25 , 0.    , 0.75 ]])
        group_indexes = [[4, 5, 6, 7], [], [0, 1, 2], [3], [12, 14], [8, 10], [13, 15], [9, 11]]
        markov_chain = get_markov_chain(compressed_transition_matrix, group_indexes)
        all_paths = get_all_paths(markov_chain)
        self.assertEqual(all_paths, [('0', '2'), ('0', '3'), ('0', '4'), ('0', '6'), ('4', '5'), ('6', '4'), ('6', '7'), ('7', '5'),
                                     ('0', '4', '5'), ('0', '6', '4'), ('0', '6', '7'), ('6', '4', '5'), ('6', '7', '5'),
                                     ('0', '6', '4', '5'), ('0', '6', '7', '5')])


class TestGetMarkovChainPathProbs(unittest.TestCase):
    def test_simple_markov_chain(self):
        markov_chain = nx.DiGraph()
        markov_chain.add_edge('A', 'A', weight=0.5)
        markov_chain.add_edge('A', 'B', weight=0.5)
        markov_chain.add_edge('B', 'B', weight=0.5)
        markov_chain.add_edge('B', 'C', weight=0.5)
        path_probabilities = get_markov_chain_path_probs(markov_chain)
        expected_result = {('A', 'B'): 1, ('B', 'C'): 1, ('A', 'B', 'C'): 1}
        self.assertEqual(path_probabilities, expected_result)

    def test_simple_markov_chain_with_paths(self):
        markov_chain = nx.DiGraph()
        markov_chain.add_edge('A', 'A', weight=0.5)
        markov_chain.add_edge('A', 'B', weight=0.5)
        markov_chain.add_edge('B', 'B', weight=0.5)
        markov_chain.add_edge('B', 'C', weight=0.5)
        all_paths = get_all_paths(markov_chain)
        path_probabilities = get_markov_chain_path_probs(markov_chain, all_paths)
        expected_result = {('A', 'B'): 1, ('B', 'C'): 1, ('A', 'B', 'C'): 1}
        self.assertEqual(path_probabilities, expected_result)

    def test_markov_chain_with_multiple_paths(self):
        markov_chain = nx.DiGraph()
        markov_chain.add_edge('A', 'B', weight=0.5)
        markov_chain.add_edge('A', 'C', weight=0.5)
        markov_chain.add_edge('B', 'B', weight=0.5)
        markov_chain.add_edge('B', 'D', weight=0.5)
        markov_chain.add_edge('C', 'C', weight=0.5)
        markov_chain.add_edge('C', 'D', weight=0.5)
        path_probabilities = get_markov_chain_path_probs(markov_chain)
        expected_result = {('A', 'B'): 0.5, ('A', 'C'): 0.5, ('B', 'D'): 1, ('C', 'D'): 1, ('A', 'B', 'D'): 0.5, ('A', 'C', 'D'): 0.5}
        self.assertEqual(path_probabilities, expected_result)

    def test_markov_chain_with_cycle(self):
        markov_chain = nx.DiGraph()
        markov_chain.add_edge('A', 'A', weight=0.5)
        markov_chain.add_edge('A', 'B', weight=0.5)
        markov_chain.add_edge('B', 'B', weight=0.5)
        markov_chain.add_edge('B', 'C', weight=0.5)
        markov_chain.add_edge('C', 'C', weight=0.5)
        markov_chain.add_edge('C', 'A', weight=0.5)
        path_probabilities = get_markov_chain_path_probs(markov_chain)
        expected_result = {('A', 'B'): 1, ('B', 'C'): 1, ('C', 'A'): 1,
                           ('A', 'B', 'C'): 1, ('B', 'C', 'A'): 1, ('C', 'A', 'B'): 1}
        self.assertEqual(path_probabilities, expected_result)

    def test_empty_markov_chain(self):
        markov_chain = nx.DiGraph()
        path_probabilities = get_markov_chain_path_probs(markov_chain)
        expected_result = {}
        self.assertEqual(path_probabilities, expected_result)

    def test_markov_chain_with_no_edges(self):
        markov_chain = nx.DiGraph()
        markov_chain.add_node('A')
        markov_chain.add_node('B')
        path_probabilities = get_markov_chain_path_probs(markov_chain)
        expected_result = {}
        self.assertEqual(path_probabilities, expected_result)

    def test_markov_chain_with_zero_weight_edges(self):
        markov_chain = nx.DiGraph()
        markov_chain.add_edge('A', 'A', weight=1)
        markov_chain.add_edge('A', 'B', weight=0)
        markov_chain.add_edge('B', 'B', weight=1)
        markov_chain.add_edge('B', 'C', weight=0)
        with self.assertRaises(ValueError):
            get_markov_chain_path_probs(markov_chain, DEBUG=True)

    def test_example(self):
        compressed_transition_matrix = np.array([[0.75 , 0.0625, 0.0625, 0.0625, 0.   , 0.0625, 0.   ],
                                                 [0.   , 1.    , 0.    , 0.    , 0.   , 0.    , 0.   ],
                                                 [0.   , 0.    , 1.    , 0.    , 0.   , 0.    , 0.   ],
                                                 [0.   , 0.    , 0.    , 0.875 , 0.125, 0.    , 0.   ],
                                                 [0.   , 0.    , 0.    , 0.    , 1.   , 0.    , 0.   ],
                                                 [0.   , 0.    , 0.    , 0.25  , 0.   , 0.625 , 0.125],
                                                 [0.   , 0.    , 0.    , 0.    , 0.25 , 0.    , 0.75 ]])
        group_indexes = [[4, 5, 6, 7], [], [0, 1, 2], [3], [12, 14], [8, 10], [13, 15], [9, 11]]
        markov_chain = get_markov_chain(compressed_transition_matrix, group_indexes)
        all_paths = get_all_paths(markov_chain)
        path_probabilities = get_markov_chain_path_probs(markov_chain, all_paths)
        expected_result = {('0', '2'): np.float64(1/4), ('0', '3'): np.float64(1/4), ('0', '4'): np.float64(1/4), ('0', '6'): np.float64(1/4), ('4', '5'): np.float64(1), ('6', '4'): np.float64(2/3), ('6', '7'): np.float64(1/3), ('7', '5'): np.float64(1),
                           ('0', '4', '5'): np.float64(1/4), ('0', '6', '4'): np.float64(1/6), ('0', '6', '7'): np.float64(1/12), ('6', '4', '5'): np.float64(2/3), ('6', '7', '5'): np.float64(1/3),
                           ('0', '6', '4', '5'): np.float64(1/6), ('0', '6', '7', '5'): np.float64(1/12)}
        self.assertEqual(path_probabilities, expected_result)


class TestGetSTGPathProbs(unittest.TestCase):

    def test_invalid_input(self):
        # Call the function with missing STG and transition matrix
        with self.assertRaises(ValueError):
            get_stg_path_probs([(0, 1)], [[0], [1]], None, None)
    def test_empty_stg_and_transition_matrix(self):
        # Create an empty STG and transition matrix
        stg = nx.DiGraph()
        transition_matrix = np.array([])
        group_indices = []
        all_paths = []
        # Call the function
        path_probabilities = get_stg_path_probs(all_paths, group_indices, stg, transition_matrix)
        # Check the output
        self.assertEqual(path_probabilities, {})

    def test_example(self):
        transition_matrix = np.array([[0, 1/2, 1/2,   0,   0,   0,   0,   0,   0,   0],
                                      [0,   0, 1/3,   0, 1/3, 1/3,   0,   0,   0,   0],
                                      [0,   0,   0,   1,   0,   0,   0,   0,   0,   0],
                                      [0,   0, 1/2,   0,   0,   0, 1/2,   0,   0,   0],
                                      [0,   0,   0,   0,   0, 1/3,   0, 1/3, 1/3,   0],
                                      [0,   0,   0,   0,   1,   0,   0,   0,   0,   0],
                                      [0,   0,   0,   0,   0,   0,   0,   1,   0,   0],
                                      [0,   0,   0,   0,   0,   0,   1,   0,   0,   0],
                                      [0,   0,   0,   0,   0,   0,   0,   0,   0,   1],
                                      [0,   0,   0,   0,   0,   0,   0,   0,   1,   0]])
        group_indices = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
        all_paths = [('0', '1'), ('0', '2'), ('1', '3'), ('2', '3'), ('2', '4'),
                     ('0', '1', '3'), ('0', '2', '3'), ('0', '2', '4')]
        path_probabilities = get_stg_path_probs(all_paths, group_indices, None, transition_matrix)
        expected_result = {('0', '1'): np.float64(1/2), ('0', '2'): np.float64(1/2), ('1', '3'): np.float64(1), ('2', '3'): np.float64(1/2), ('2', '4'): np.float64(1/2),
                           ('0', '1', '3'): np.float64(1/2), ('0', '2', '3'): np.float64(1/4), ('0', '2', '4'): np.float64(1/4)}
        self.assertTrue(np.allclose(list(path_probabilities.values()), list(expected_result.values())))

class TestSolveMatrixEquation(unittest.TestCase):
    def test_non_singular_matrix(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[1], [2]])
        X = solve_matrix_equation(A, B)
        self.assertIsNotNone(X)
        self.assertIsInstance(X, np.ndarray)
    def test_singular_matrix_no_solution(self):
        A = np.array([[1, 0], [0, 1]])
        B = np.array([[1], [2]])
        X = solve_matrix_equation(A, B)
        self.assertIsNone(X)
    def test_singular_matrix_infinite_solutions(self):
        A = np.array([[1, 0], [0, 1]])
        B = np.array([[0], [0]])
        X = solve_matrix_equation(A, B)
        self.assertIsNone(X)
    def test_non_square_matrix(self):
        A = np.array([[1, 2, 3], [4, 5, 6]])
        B = np.array([[1], [2]])
        with self.assertRaises(ValueError):
            solve_matrix_equation(A, B, DEBUG=True)
    def test_example(self):
        A = np.array([[0, 1/2], [0, 0]])
        B = np.array([[1/2, 0], [0, 1/3]])
        X = solve_matrix_equation(A, B)
        expected_result = np.array([[1/2, 1/6], [0, 1/3]])
        self.assertTrue(np.allclose(X, expected_result))


class TestComparePathReachability(unittest.TestCase):
    def test_valid_input(self):
        stg_probabilities = {(1, 2): 0.5, (2, 3): 0.3, (1, 2, 3): 0}
        markov_probabilities = {(1, 2): 0.4, (2, 3): 0.3, (1, 2, 3): 0.1}
        TP, FP, TN, FN = compare_path_reachability(stg_probabilities, markov_probabilities)
        self.assertEqual(TP, 2)
        self.assertEqual(FP, 1)
        self.assertEqual(TN, 0)
        self.assertEqual(FN, 0)

    def test_edges_comparison(self):
        stg_probabilities = {(1, 2): 0.5, (2, 3): 0.3, (1, 2, 3): 0}
        markov_probabilities = {(1, 2): 0.4, (2, 3): 0.3, (1, 2, 3): 0.1}
        TP, FP, TN, FN = compare_path_reachability(stg_probabilities, markov_probabilities, type="edges")
        self.assertEqual(TP, 2)
        self.assertEqual(FP, 0)
        self.assertEqual(TN, 0)
        self.assertEqual(FN, 0)

    def test_non_edges_comparison(self):
        stg_probabilities = {(1, 2): 0.5, (2, 3): 0.3, (1, 2, 3): 0}
        markov_probabilities = {(1, 2): 0.4, (2, 3): 0.3, (1, 2, 3): 0.1}
        TP, FP, TN, FN = compare_path_reachability(stg_probabilities, markov_probabilities, type="non_edges")
        self.assertEqual(TP, 0)
        self.assertEqual(FP, 1)
        self.assertEqual(TN, 0)
        self.assertEqual(FN, 0)

    def test_path_i_comparison(self):
        stg_probabilities = {(1, 2): 0.5, (2, 3): 0.3, (1, 2, 3): 0}
        markov_probabilities = {(1, 2): 0.4, (2, 3): 0.3, (1, 2, 3): 0.1}
        TP, FP, TN, FN = compare_path_reachability(stg_probabilities, markov_probabilities, type="path_2")
        self.assertEqual(TP, 0)
        self.assertEqual(FP, 1)
        self.assertEqual(TN, 0)
        self.assertEqual(FN, 0)

    def test_debug_mode(self):
        stg_probabilities = {(1, 2): 0.5, (2, 3): 0.3}
        markov_probabilities = {(1, 2): 0.4, (3, 4): 0.1}
        with self.assertRaises(ValueError):
            compare_path_reachability(stg_probabilities, markov_probabilities, DEBUG=True)


class TestComparePathRmsd(unittest.TestCase):
    def test_valid_input(self):
        stg_probabilities = {(1, 2): 0.5, (2, 3): 0.3, (1, 2, 3): 0}
        markov_probabilities = {(1, 2): 0.4, (2, 3): 0.3, (1, 2, 3): 0.1}
        rmsd = compare_path_rmsd(stg_probabilities, markov_probabilities)
        self.assertAlmostEqual(rmsd, 0.0816497)

    def test_edges_comparison(self):
        stg_probabilities = {(1, 2): 0.5, (2, 3): 0.3, (1, 2, 3): 0}
        markov_probabilities = {(1, 2): 0.4, (2, 3): 0.3, (1, 2, 3): 0.1}
        rmsd = compare_path_rmsd(stg_probabilities, markov_probabilities, type="edges")
        self.assertAlmostEqual(rmsd, 0.0707107)

    def test_non_edges_comparison(self):
        stg_probabilities = {(1, 2): 0.5, (2, 3): 0.3, (1, 2, 3): 0}
        markov_probabilities = {(1, 2): 0.4, (2, 3): 0.3, (1, 2, 3): 0.1}
        rmsd = compare_path_rmsd(stg_probabilities, markov_probabilities, type="non_edges")
        self.assertAlmostEqual(rmsd, 0.1)

    def test_path_i_comparison(self):
        stg_probabilities = {(1, 2): 0.5, (2, 3): 0.3, (1, 2, 3): 0}
        markov_probabilities = {(1, 2): 0.4, (2, 3): 0.3, (1, 2, 3): 0.1}
        rmsd = compare_path_rmsd(stg_probabilities, markov_probabilities, type="path_2")
        self.assertAlmostEqual(rmsd, 0.1)

    def test_debug_mode(self):
        stg_probabilities = {(1, 2): 0.5, (2, 3): 0.3}
        markov_probabilities = {(1, 2): 0.4, (3, 4): 0.1}
        with self.assertRaises(ValueError):
            compare_path_rmsd(stg_probabilities, markov_probabilities, DEBUG=True)

if __name__ == '__main__':
    unittest.main()