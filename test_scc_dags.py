import unittest

import networkx as nx
from pyboolnet.external.bnet2primes import bnet_text2primes
from pyboolnet.state_transition_graphs import primes2stg

from scc_dags import get_scc_dag, get_ordered_states, get_attractor_states

class TestGetSCCDAG(unittest.TestCase):
    def test_empty_graph(self):
        stg = nx.DiGraph()
        dag = get_scc_dag(stg)
        self.assertEqual(dag.number_of_nodes(), 0)
        self.assertEqual(dag.number_of_edges(), 0)

    def test_single_scc(self):
        stg = nx.DiGraph()
        stg.add_edge(1, 2)
        stg.add_edge(2, 1)
        dag = get_scc_dag(stg)
        self.assertEqual(dag.number_of_nodes(), 1)
        self.assertEqual(dag.number_of_edges(), 0)

    def test_multiple_sccs(self):
        stg = nx.DiGraph()
        stg.add_edge(1, 2)
        stg.add_edge(2, 1)
        stg.add_edge(3, 4)
        stg.add_edge(4, 3)
        stg.add_edge(1, 3)
        dag = get_scc_dag(stg)
        self.assertEqual(dag.number_of_nodes(), 2)
        self.assertEqual(dag.number_of_edges(), 1)

    def test_self_loops(self):
        stg = nx.DiGraph()
        stg.add_edge(1, 1)
        stg.add_edge(1, 2)
        stg.add_edge(2, 1)
        dag = get_scc_dag(stg)
        self.assertEqual(dag.number_of_nodes(), 1)
        self.assertEqual(dag.number_of_edges(), 0)

    def test_parallel_edges(self):
        stg = nx.DiGraph()
        stg.add_edge(1, 2)
        stg.add_edge(1, 2)
        stg.add_edge(2, 1)
        dag = get_scc_dag(stg)
        self.assertEqual(dag.number_of_nodes(), 1)
        self.assertEqual(dag.number_of_edges(), 0)

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

        dag = get_scc_dag(stg)

        self.assertEqual(dag.number_of_nodes(), 9)
        self.assertEqual(dag.number_of_edges(), 10)


class TestGetOrderedStates(unittest.TestCase):
    def test_simple_scc_dag(self):
        # Create a simple SCC DAG with one node
        scc_dag = nx.DiGraph()
        scc_dag.add_node(0, states=['000'])
        ordered_states = get_ordered_states(scc_dag)
        self.assertEqual(ordered_states, [['000']])

    def test_multiple_nodes_and_edges(self):
        # Create a SCC DAG with multiple nodes and edges
        scc_dag = nx.DiGraph()
        scc_dag.add_node(0, states=['000', '001'])
        scc_dag.add_node(1, states=['010', '011'])
        scc_dag.add_edge(0, 1)
        ordered_states = get_ordered_states(scc_dag)
        self.assertEqual(ordered_states, [['000', '001'], ['010', '011']])

    def test_multiple_strongly_connected_components(self):
        # Create a SCC DAG with multiple strongly connected components
        scc_dag = nx.DiGraph()
        scc_dag.add_node(0, states=['000', '001'])
        scc_dag.add_node(1, states=['010', '011'])
        scc_dag.add_node(2, states=['100', '101'])
        scc_dag.add_edge(0, 1)
        scc_dag.add_edge(1, 2)
        ordered_states = get_ordered_states(scc_dag)
        self.assertEqual(ordered_states, [['000', '001'], ['010', '011'], ['100', '101']])

    def test_as_indexes_true(self):
        # Test with `as_indexes=True` to get decimal indexes
        scc_dag = nx.DiGraph()
        scc_dag.add_node(0, states=['000', '001'])
        ordered_states = get_ordered_states(scc_dag, as_indexes=True)
        self.assertEqual(ordered_states, [[0, 1]])

    def test_as_indexes_false(self):
        # Test with `as_indexes=False` to get binary strings
        scc_dag = nx.DiGraph()
        scc_dag.add_node(0, states=['000', '001'])
        ordered_states = get_ordered_states(scc_dag, as_indexes=False)
        self.assertEqual(ordered_states, [['000', '001']])

    def test_empty_scc_dag(self):
        # Test with an empty SCC DAG
        scc_dag = nx.DiGraph()
        ordered_states = get_ordered_states(scc_dag)
        self.assertEqual(ordered_states, [])

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

        scc_dag = get_scc_dag(stg)

        ordered_states = get_ordered_states(scc_dag)
        ordered_indexes = get_ordered_states(scc_dag, as_indexes=True)

        self.assertEqual(ordered_states, [['1001'], ['0101', '0111'], ['1101', '1111'], ['0100', '0110'], ['1011'], ['1100', '1110'], ['1000', '1010'], ['0011'], ['0000', '0001', '0010']])
        self.assertEqual(ordered_indexes, [[9], [5, 7], [13, 15], [4, 6], [11], [12, 14], [8, 10], [3], [0, 1, 2]])


class TestGetAttractorStates(unittest.TestCase):
    def test_simple_scc_dag(self):
        # Create a simple SCC DAG with one attractor state
        scc_dag = nx.DiGraph()
        scc_dag.add_node(0, states=['101'])
        scc_dag.add_node(1, states=['110'])
        scc_dag.add_edge(0, 1)

        attractor_states = get_attractor_states(scc_dag)
        self.assertEqual(attractor_states, [['110']])

    def test_multiple_attractor_states(self):
        # Create a SCC DAG with multiple attractor states
        scc_dag = nx.DiGraph()
        scc_dag.add_node(0, states=['101'])
        scc_dag.add_node(1, states=['110'])
        scc_dag.add_node(2, states=['111'])
        scc_dag.add_edge(0, 1)
        scc_dag.add_edge(1, 2)

        attractor_states = get_attractor_states(scc_dag)
        self.assertEqual(attractor_states, [['111']])

    def test_no_attractor_states(self):
        # Create a SCC DAG with no attractor states
        scc_dag = nx.DiGraph()
        scc_dag.add_node(0, states=['101'])
        scc_dag.add_node(1, states=['110'])
        scc_dag.add_edge(0, 1)
        scc_dag.add_edge(1, 0)

        with self.assertRaises(ValueError):
            get_attractor_states(scc_dag, DEBUG=True)

    def test_as_indexes_true(self):
        # Create a SCC DAG and test with `as_indexes=True`
        scc_dag = nx.DiGraph()
        scc_dag.add_node(0, states=['101'])
        scc_dag.add_node(1, states=['110'])
        scc_dag.add_edge(0, 1)

        attractor_states = get_attractor_states(scc_dag, as_indexes=True)
        self.assertEqual(attractor_states, [[6]])

    def test_as_indexes_false(self):
        # Create a SCC DAG and test with `as_indexes=False`
        scc_dag = nx.DiGraph()
        scc_dag.add_node(0, states=['101'])
        scc_dag.add_node(1, states=['110'])
        scc_dag.add_edge(0, 1)

        attractor_states = get_attractor_states(scc_dag, as_indexes=False)
        self.assertEqual(attractor_states, [['110']])

    def test_empty_scc_dag(self):
        # Create an empty SCC DAG
        scc_dag = nx.DiGraph()

        attractor_states = get_attractor_states(scc_dag)
        self.assertEqual(attractor_states, [])

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

        scc_dag = get_scc_dag(stg)

        attractor_states = get_attractor_states(scc_dag)
        attractor_indexes = get_attractor_states(scc_dag, as_indexes=True)

        self.assertEqual(attractor_states, [['1000', '1010'], ['0011'], ['0000', '0001', '0010']])
        self.assertEqual(attractor_indexes, [[8, 10], [3], [0, 1, 2]])

if __name__ == '__main__':
    unittest.main()