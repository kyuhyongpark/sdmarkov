import unittest

import networkx as nx
from pyboolnet.external.bnet2primes import bnet_text2primes
from pyboolnet.state_transition_graphs import primes2stg

from scc_dags import get_scc_dag, get_scc_states, get_attractor_states

class TestGetSCCDAG(unittest.TestCase):
    def test_empty_graph(self):
        stg = nx.DiGraph()
        scc_dag = get_scc_dag(stg)
        self.assertEqual(scc_dag.number_of_nodes(), 0)
        self.assertEqual(scc_dag.number_of_edges(), 0)

    def test_single_scc(self):
        stg = nx.DiGraph()
        stg.add_edge('0', '1')
        stg.add_edge('1', '0')
        scc_dag = get_scc_dag(stg)
        self.assertEqual(scc_dag.number_of_nodes(), 1)
        self.assertEqual(scc_dag.number_of_edges(), 0)

    def test_multiple_sccs(self):
        stg = nx.DiGraph()
        stg.add_edge('00', '01')
        stg.add_edge('01', '00')
        stg.add_edge('10', '11')
        stg.add_edge('11', '10')
        stg.add_edge('00', '10')
        scc_dag = get_scc_dag(stg)
        self.assertEqual(scc_dag.number_of_nodes(), 2)
        self.assertEqual(scc_dag.number_of_edges(), 1)

    def test_self_loops(self):
        stg = nx.DiGraph()
        stg.add_edge('0', '0')
        stg.add_edge('0', '1')
        stg.add_edge('1', '0')
        scc_dag = get_scc_dag(stg)
        self.assertEqual(scc_dag.number_of_nodes(), 1)
        self.assertEqual(scc_dag.number_of_edges(), 0)

    def test_parallel_edges(self):
        stg = nx.DiGraph()
        stg.add_edge('0', '1')
        stg.add_edge('0', '1')
        stg.add_edge('1', '0')
        scc_dag = get_scc_dag(stg)
        self.assertEqual(scc_dag.number_of_nodes(), 1)
        self.assertEqual(scc_dag.number_of_edges(), 0)

    def test_markov_chain(self):
        markov_chain = nx.DiGraph()
        markov_chain.add_node('G0', states=['00', '01'])
        markov_chain.add_node('G2', states=['10', '11'])
        markov_chain.add_edge('G0', 'G0', weight=0.5)
        markov_chain.add_edge('G0', 'G2', weight=0.5)
        markov_chain.add_edge('G2', 'G0', weight=0.3)
        markov_chain.add_edge('G2', 'G2', weight=0.7)

        scc_dag = get_scc_dag(markov_chain)

        self.assertEqual(scc_dag.number_of_nodes(), 1)
        self.assertEqual(scc_dag.nodes[0]['states'], ['00', '01', '10', '11'])
        self.assertEqual(scc_dag.number_of_edges(), 0)

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


class TestGetSCCStates(unittest.TestCase):
    def test_simple_scc_dag(self):
        # Create a simple SCC DAG with one node
        scc_dag = nx.DiGraph()
        scc_dag.add_node(0, states=['000'])
        scc_states = get_scc_states(scc_dag)
        self.assertEqual(scc_states, [['000']])

    def test_multiple_nodes_and_edges(self):
        # Create a SCC DAG with multiple nodes and edges
        scc_dag = nx.DiGraph()
        scc_dag.add_node(0, states=['000', '001'])
        scc_dag.add_node(1, states=['010', '011'])
        scc_dag.add_edge(0, 1)
        scc_states = get_scc_states(scc_dag)
        self.assertEqual(scc_states, [['000', '001'], ['010', '011']])

    def test_multiple_strongly_connected_components(self):
        # Create a SCC DAG with multiple strongly connected components
        scc_dag = nx.DiGraph()
        scc_dag.add_node(0, states=['000', '001'])
        scc_dag.add_node(1, states=['010', '011'])
        scc_dag.add_node(2, states=['100', '101'])
        scc_dag.add_edge(0, 1)
        scc_dag.add_edge(1, 2)
        scc_states = get_scc_states(scc_dag)
        self.assertEqual(scc_states, [['000', '001'], ['010', '011'], ['100', '101']])

    def test_as_indices_true(self):
        # Test with `as_indices=True` to get decimal indices
        scc_dag = nx.DiGraph()
        scc_dag.add_node(0, states=['000', '001'])
        scc_states = get_scc_states(scc_dag, as_indices=True)
        self.assertEqual(scc_states, [[0, 1]])

    def test_as_indices_false(self):
        # Test with `as_indices=False` to get binary strings
        scc_dag = nx.DiGraph()
        scc_dag.add_node(0, states=['000', '001'])
        scc_states = get_scc_states(scc_dag, as_indices=False)
        self.assertEqual(scc_states, [['000', '001']])

    def test_empty_scc_dag(self):
        # Test with an empty SCC DAG
        scc_dag = nx.DiGraph()
        scc_states = get_scc_states(scc_dag)
        self.assertEqual(scc_states, [])

    def test_markov_chain(self):
        markov_chain = nx.DiGraph()
        markov_chain.add_node('G0', states=['00', '01'])
        markov_chain.add_node('G2', states=['10', '11'])
        markov_chain.add_edge('G0', 'G0', weight=0.5)
        markov_chain.add_edge('G0', 'G2', weight=0.5)
        markov_chain.add_edge('G2', 'G0', weight=0.3)
        markov_chain.add_edge('G2', 'G2', weight=0.7)

        scc_dag = get_scc_dag(markov_chain)

        scc_states = get_scc_states(scc_dag)
        self.assertEqual(scc_states, [['00', '01', '10', '11']])

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

        scc_states = get_scc_states(scc_dag)
        scc_indices = get_scc_states(scc_dag, as_indices=True)

        self.assertEqual(scc_states, [['1001'], ['0101', '0111'], ['1101', '1111'], ['0100', '0110'], ['1011'], ['1100', '1110'], ['0000', '0001', '0010'], ['0011'], ['1000', '1010']])
        self.assertEqual(scc_indices, [[9], [5, 7], [13, 15], [4, 6], [11], [12, 14], [0, 1, 2], [3], [8, 10]])


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

    def test_as_indices_true(self):
        # Create a SCC DAG and test with `as_indices=True`
        scc_dag = nx.DiGraph()
        scc_dag.add_node(0, states=['101'])
        scc_dag.add_node(1, states=['110'])
        scc_dag.add_edge(0, 1)

        attractor_states = get_attractor_states(scc_dag, as_indices=True)
        self.assertEqual(attractor_states, [[6]])

    def test_as_indices_false(self):
        # Create a SCC DAG and test with `as_indices=False`
        scc_dag = nx.DiGraph()
        scc_dag.add_node(0, states=['101'])
        scc_dag.add_node(1, states=['110'])
        scc_dag.add_edge(0, 1)

        attractor_states = get_attractor_states(scc_dag, as_indices=False)
        self.assertEqual(attractor_states, [['110']])

    def test_empty_scc_dag(self):
        # Create an empty SCC DAG
        scc_dag = nx.DiGraph()

        attractor_states = get_attractor_states(scc_dag)
        self.assertEqual(attractor_states, [])

    def test_markov_chain(self):
        markov_chain = nx.DiGraph()
        markov_chain.add_node('G0', states=['00', '01'])
        markov_chain.add_node('G2', states=['10', '11'])
        markov_chain.add_edge('G0', 'G0', weight=0.5)
        markov_chain.add_edge('G0', 'G2', weight=0.5)
        markov_chain.add_edge('G2', 'G2', weight=1.0)

        scc_dag = get_scc_dag(markov_chain)

        attractor_states = get_attractor_states(scc_dag)
        self.assertEqual(attractor_states, [['10', '11']])

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
        attractor_indices = get_attractor_states(scc_dag, as_indices=True)

        self.assertEqual(attractor_states, [['0000', '0001', '0010'], ['0011'], ['1000', '1010']])
        self.assertEqual(attractor_indices, [[0, 1, 2], [3], [8, 10]])

if __name__ == '__main__':
    unittest.main()