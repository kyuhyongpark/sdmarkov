import unittest

from succession_diagram import get_sd_nodes_and_edges
from succession_diagram import sort_sd_nodes
from succession_diagram import generate_states
from succession_diagram import get_binary_states
from succession_diagram import get_sd_group_states
from succession_diagram import states_to_indices


class TestSortSdNodes(unittest.TestCase):
    def test_sorting_with_absent_keys(self):
        nodes = ['A', 'B', 'C', 'D']
        sd_nodes = [{'A': 0, 'B': 0}, {'A':0}, {'A': 0, 'B': 0, 'C': 0}, {}]
        expected_result = [{}, {'A':0}, {'A': 0, 'B': 0}, {'A': 0, 'B': 0, 'C': 0}]
        self.assertEqual(sort_sd_nodes(nodes, sd_nodes), expected_result)

    def test_sorting_with_values_0_and_1(self):
        nodes = ['A', 'B']
        sd_nodes = [{'A': 0, 'B': 1}, {'A': 1, 'B': 1}, {'A': 1, 'B': 0}, {'A': 0, 'B': 0}]
        expected_result = [{'A': 0, 'B': 0}, {'A': 0, 'B': 1}, {'A': 1, 'B': 0}, {'A': 1, 'B': 1}]
        self.assertEqual(sort_sd_nodes(nodes, sd_nodes), expected_result)

    def test_sorting_with_multiple_nodes_having_same_value(self):
        nodes = ['A', 'B', 'C', 'D']
        sd_nodes = [{'A': 1, 'B': 1}, {'A': 1, 'C': 1}, {'A': 1, 'D': 1}, {'B': 1, 'C': 1}]
        expected_result = [{'B': 1, 'C': 1}, {'A': 1, 'D': 1}, {'A': 1, 'C': 1}, {'A': 1, 'B': 1}]
        self.assertEqual(sort_sd_nodes(nodes, sd_nodes), expected_result)

    def test_sorting_with_empty_list_of_nodes(self):
        nodes = []
        sd_nodes = [{'A': 1, 'B': 0}, {'A': 0, 'B': 1}]
        with self.assertRaises(ValueError):
            sort_sd_nodes(nodes, sd_nodes, DEBUG=True)

    def test_handling_of_invalid_values_in_sd_nodes(self):
        nodes = ['A', 'B', 'C', 'D']
        sd_nodes = [{'A': 2, 'B': 0}, {'A': 1, 'B': 1}]
        with self.assertRaises(ValueError):
            sort_sd_nodes(nodes, sd_nodes)

    def test_handling_of_duplicates(self):
        nodes = ['A', 'B', 'C', 'D']
        sd_nodes = [{'A': 0, 'B': 0}, {'A': 0, 'B': 0}]
        with self.assertRaises(ValueError):
            sort_sd_nodes(nodes, sd_nodes, DEBUG=True)


class TestGetSdNodesAndEdges(unittest.TestCase):
    def test_valid_bnet(self):
        bnet = """
        A, A | B & C
        B, B & !C
        C, B & !C | !C & !D | !B & C & D
        D, !A & !B & !C & !D | !A & C & D
        """
        expected_output = (['A', 'B', 'C', 'D'],
                           [{},
                            {'B': 0},
                            {'A': 0, 'B': 0},
                            {'A': 0, 'B': 0, 'C': 1, 'D': 1},
                            {'A': 1, 'D': 0},
                            {'A': 1, 'B': 0, 'D': 0},],
                           [[{'A': 1}],
                            [{'A': 1, 'B': 0}]])

        self.assertEqual(get_sd_nodes_and_edges(bnet), expected_output)

    def test_multi_motif_edge(self):
        bnet = """
        A, A | B
        B, A | B
        C, A & B 
        """
        expected_output = (['A', 'B', 'C'],
                           [{},
                            {'A': 0, 'B': 0, 'C': 0},
                            {'A': 1, 'B': 1, 'C': 1},],
                           [[{'B': 1}, {'A': 1}],
                            [{'A': 0, 'B': 0}],])

        self.assertEqual(get_sd_nodes_and_edges(bnet), expected_output)

    def test_source_nodes(self):
        bnet = """
        A, A
        B, B
        C, A & B 
        """
        expected_output = (['A', 'B', 'C'],
                           [{},
                            {'A': 0, 'B': 0, 'C': 0},
                            {'A': 0, 'B': 1, 'C': 0},
                            {'A': 1, 'B': 0, 'C': 0},
                            {'A': 1, 'B': 1, 'C': 1}],
                           [[{'A': 0, 'B': 0}],
                            [{'A': 0, 'B': 1}],
                            [{'A': 1, 'B': 0}],
                            [{'A': 1, 'B': 1}]])
        self.assertEqual(get_sd_nodes_and_edges(bnet), expected_output)

    def test_duplicate_states(self):
        bnet = """
        A, A & D
        B, A & B
        C, B
        D, A
        """
        expected_output = (['A', 'B', 'C', 'D'],
                           [{},
                            {'B': 0, 'C': 0},
                            {'A': 0, 'B': 0, 'C': 0, 'D': 0},
                            {'A': 1, 'D': 1},
                            {'A': 1, 'B': 0, 'C': 0, 'D': 1},
                            {'A': 1, 'B': 1, 'C': 1, 'D': 1},],
                           [[{'B': 0},],
                            [{'A': 0},],
                            [{'A': 0, 'B': 0, 'C': 0},],
                            [{'A': 1, 'B': 0, 'D': 1},],
                            [{'A': 1, 'B': 1, 'D': 1},],])

        self.assertEqual(get_sd_nodes_and_edges(bnet), expected_output)

    def test_percolation(self):
        bnet = """
        X1, X1 | X2 | X3
        X2, X1
        X3, X2
        Y1, Y1 | Y2 | Y3
        Y2, Y1
        Y3, Y2
        """
        expected_output = (['X1', 'X2', 'X3', 'Y1', 'Y2', 'Y3'],
                           [{},
                            {'Y1':0, 'Y2':0, 'Y3':0},
                            {'Y1':1, 'Y2':1, 'Y3':1},
                            {'X1':0, 'X2':0, 'X3':0},
                            {'X1':0, 'X2':0, 'X3':0, 'Y1':0, 'Y2':0, 'Y3':0},
                            {'X1':0, 'X2':0, 'X3':0, 'Y1':1, 'Y2':1, 'Y3':1},
                            {'X1':1, 'X2':1, 'X3':1},
                            {'X1':1, 'X2':1, 'X3':1, 'Y1':0, 'Y2':0, 'Y3':0},
                            {'X1':1, 'X2':1, 'X3':1, 'Y1':1, 'Y2':1, 'Y3':1}],
                           [[{'Y1':1}],
                            [{'X1':0, 'X2':0, 'X3':0, 'Y1':1}],
                            [{'X1':1}],
                            [{'X1':1, 'Y1':0, 'Y2':0, 'Y3':0}],
                            [{'X1':1, 'Y1':1, 'Y2':1, 'Y3':1}],
                            [{'X1':1, 'X2':1, 'X3':1, 'Y1':1}]])

        self.assertEqual(get_sd_nodes_and_edges(bnet), expected_output)


    def test_valid_bnet_minimal(self):
        bnet = """
        A, A | B & C
        B, B & !C
        C, B & !C | !C & !D | !B & C & D
        D, !A & !B & !C & !D | !A & C & D
        """
        expected_output = (['A', 'B', 'C', 'D'],
                           [{'A': 0, 'B': 0, 'C': 1, 'D': 1},
                            {'A': 1, 'B': 0, 'D': 0},],
                           [])

        self.assertEqual(get_sd_nodes_and_edges(bnet, minimal=True), expected_output)

    def test_single_node_bnet(self):
        bnet = "A, A"
        expected_output = (['A'], [{}, {'A': 0}, {'A': 1}], [])
        self.assertEqual(get_sd_nodes_and_edges(bnet), expected_output)

    def test_duplicate_node_bnet(self):
        bnet = "A, A | B & C \n B, B & !C \n A, A | B & C"
        with self.assertRaises(Exception):
            get_sd_nodes_and_edges(bnet)

    def test_invalid_bnet(self):
        bnet = "this is not a valid bnet"
        with self.assertRaises(Exception):
            get_sd_nodes_and_edges(bnet)


class TestGenerateStates(unittest.TestCase):
    def test_get_all_states(self):
        nodes = ['A', 'B', 'C']
        node_values = {}
        valid_exclude_values = []
        expected_result = ['000', '001', '010', '011', '100', '101', '110', '111']
        self.assertEqual(generate_states(nodes, node_values, valid_exclude_values), expected_result)

    def test_fixed_node_values(self):
        nodes = ['A', 'B', 'C']
        node_values = {'A': 1, 'B': 0}
        valid_exclude_values = []
        expected_result = ['100', '101']
        self.assertEqual(generate_states(nodes, node_values, valid_exclude_values), expected_result)

    def test_excluded_values(self):
        nodes = ['A', 'B', 'C']
        node_values = {}
        valid_exclude_values = [{'A': 1,'B': 1, 'C': 1}, {'C': 0}]
        expected_result = ['001', '011', '101']
        self.assertEqual(generate_states(nodes, node_values, valid_exclude_values), expected_result)

    def test_combined(self):
        nodes = ['A', 'B', 'C', 'D']
        node_values = {'A': 1}
        valid_exclude_values = [{'A': 1,'B': 1, 'C': 1}, {'D': 0}]
        expected_result = ['1001', '1011', '1101']
        self.assertEqual(generate_states(nodes, node_values, valid_exclude_values), expected_result)

    def test_impossible_states(self):
        nodes = ['A', 'B']
        node_values = {'A': 1}
        valid_exclude_values = [{'A': 1}, {'B': 0}]
        expected_result = []
        self.assertEqual(generate_states(nodes, node_values, valid_exclude_values), expected_result)

    def test_invalid_node_values(self):
        nodes = ['A', 'B', 'C', 'D']
        node_values = {'A': 2}
        valid_exclude_values = []
        with self.assertRaises(ValueError):
            generate_states(nodes, node_values, valid_exclude_values, DEBUG=True)

    def test_invalid_excluded_values(self):
        nodes = ['A', 'B', 'C', 'D']
        node_values = {'A': 1}
        valid_exclude_values = [{'A': 2,'B': 1, 'C': 1}]
        with self.assertRaises(ValueError):
            generate_states(nodes, node_values, valid_exclude_values, DEBUG=True)


class TestGetBinaryStates(unittest.TestCase):
    def test_no_excluded_values(self):
        nodes = ['A', 'B']
        node_values = {'A': 1}
        exclude_values_list = []
        expected_result = ['10', '11']
        self.assertEqual(get_binary_states(nodes, node_values, exclude_values_list), expected_result)

    def test_excluded_values_subset(self):
        nodes = ['A', 'B']
        node_values = {'A': 1}
        exclude_values_list = [{'A': 1, 'B': 1}, {'A': 1, 'B': 0}]
        expected_result = []
        self.assertEqual(get_binary_states(nodes, node_values, exclude_values_list), expected_result)

    def test_excluded_values_not_subset(self):
        nodes = ['A', 'B']
        node_values = {'A': 1}
        exclude_values_list = [{'A': 0, 'B': 1}, {'B': 0}]
        expected_result = ['10', '11']
        self.assertEqual(get_binary_states(nodes, node_values, exclude_values_list), expected_result)

    def test_multiple_excluded_values(self):
        nodes = ['A', 'B', 'C', 'D']
        node_values = {'A': 1}
        exclude_values_list = [{'A':1, 'B': 1, 'C':1}, {'A':1, 'C': 0}, {'A':0}, {'D': 0}]
        expected_result = ['1010', '1011']
        self.assertEqual(get_binary_states(nodes, node_values, exclude_values_list), expected_result)

    def test_node_values_multiple_nodes(self):
        nodes = ['A', 'B', 'C', 'D']
        node_values = {'A': 1, 'B': 0}
        exclude_values_list = [{'A': 1, 'B': 1}, {'A': 1, 'C': 0}, {'A': 1, 'B': 0, 'C': 0}, {}]
        expected_result = ['1010', '1011']
        self.assertEqual(get_binary_states(nodes, node_values, exclude_values_list), expected_result)

    def test_invalid_input(self):
        nodes = ['A', 'B', 'C', 'D']
        node_values = {'A': 2}
        exclude_values_list = [{'A': 1, 'B': 1}, {'A': 1, 'C': 0}]
        with self.assertRaises(ValueError):
            get_binary_states(nodes, node_values, exclude_values_list, DEBUG=True)


class TestGetSDGroupStates(unittest.TestCase):
    def test_example_from_docstring(self):
        nodes = ['A', 'B', 'C', 'D']
        sd_nodes = [{},
                    {'B': 0},
                    {'A': 0, 'B': 0},
                    {'A': 0, 'B': 0, 'C': 1, 'D': 1},
                    {'A': 1, 'D': 0},
                    {'A': 1, 'B': 0, 'D': 0},]
        sd_edges = [[{'A': 1}],
                    [{'A': 1, 'B': 0}]]
        expected_result = (
            True,
            [['0100', '0101', '0110', '0111'],
             [],
             ['0000', '0001', '0010'],
             ['0011'],
             ['1100', '1110'],
             ['1000', '1010'],
             ['1101', '1111'],
             ['1001', '1011']],
            {})
        self.assertEqual(get_sd_group_states(nodes, sd_nodes, sd_edges), expected_result)

    def test_combined(self):
        bnet="""
        A, A | B & C
        B, B & !C
        C, B & !C | !C & !D | !B & C & D
        D, !A & !B & !C & !D | !A & C & D
        """
        nodes, sd_nodes, sd_edges = get_sd_nodes_and_edges(bnet)
        expected_result = (
            True,
            [['0100', '0101', '0110', '0111'],
             [],
             ['0000', '0001', '0010'],
             ['0011'],
             ['1100', '1110'],
             ['1000', '1010'],
             ['1101', '1111'],
             ['1001', '1011']],
            {})
        self.assertEqual(get_sd_group_states(nodes, sd_nodes, sd_edges), expected_result)

    def test_minimal(self):
        bnet="""
        A, A | B & C
        B, B & !C
        C, B & !C | !C & !D | !B & C & D
        D, !A & !B & !C & !D | !A & C & D
        """
        nodes, sd_nodes, sd_edges = get_sd_nodes_and_edges(bnet, minimal=True)
        expected_result = (
            True,
            [['0011'],
             ['1000', '1010']],
            {})
        self.assertEqual(get_sd_group_states(nodes, sd_nodes, sd_edges), expected_result)

    def test_multi_motif_edge(self):
        bnet="""
        A, A | B
        B, A | B
        C, A & B 
        """
        nodes, sd_nodes, sd_edges = get_sd_nodes_and_edges(bnet)
        expected_result = (
            True,
            [[],
             ['000'],
             ['111'],
             ['010', '011', '100', '101', '110'],
             ['001']],
            {})
        self.assertEqual(get_sd_group_states(nodes, sd_nodes, sd_edges), expected_result)

    def test_duplicate_states(self):
        nodes = ['A', 'B', 'C', 'D']
        sd_nodes = [{},
                    {'B': 0, 'C': 0},
                    {'A': 0, 'B': 0, 'C': 0, 'D': 0},
                    {'A': 1, 'D': 1},
                    {'A': 1, 'B': 0, 'C': 0, 'D': 1},
                    {'A': 1, 'B': 1, 'C': 1, 'D': 1},]
        sd_edges = [[{'B': 0}],
                    [{'A': 0}],
                    [{'A': 0, 'B': 0, 'C': 0}],
                    [{'A': 1, 'B': 0, 'D': 1}],
                    [{'A': 1, 'B': 1, 'D': 1}]]
        expected_result = (
            False,
            [['1100', '1110',],
             ['1000'],
             ['0000'],
             [],
             ['1001'],
             ['1111'],
             ['0010', '0011', '1010'],
             ['0010', '0011', '0100', '0101', '0110', '0111',],
             ['0001'],
             ['1011'],
             ['1101'],],
            {'0010': [[{'B': 0}], [{'A': 0}],],
             '0011': [[{'B': 0}], [{'A': 0}],],})
        self.assertEqual(get_sd_group_states(nodes, sd_nodes, sd_edges), expected_result)

    def test_duplicate_states_combined(self):
        bnet="""
        A, A & D
        B, A & B
        C, B
        D, A
        """
        nodes, sd_nodes, sd_edges = get_sd_nodes_and_edges(bnet)
        expected_result = (
            False,
            [['1100', '1110',],
             ['1000'],
             ['0000'],
             [],
             ['1001'],
             ['1111'],
             ['0010', '0011', '1010'],
             ['0010', '0011', '0100', '0101', '0110', '0111',],
             ['0001'],
             ['1011'],
             ['1101'],],
            {'0010': [[{'B': 0}], [{'A': 0}],],
             '0011': [[{'B': 0}], [{'A': 0}],],})
        self.assertEqual(get_sd_group_states(nodes, sd_nodes, sd_edges), expected_result)

    def test_percolation(self):
        nodes = ['X1', 'X2', 'X3', 'Y1', 'Y2', 'Y3']
        sd_nodes = [{},
                    {'Y1':0, 'Y2':0, 'Y3':0},
                    {'Y1':1, 'Y2':1, 'Y3':1},
                    {'X1':0, 'X2':0, 'X3':0},
                    {'X1':0, 'X2':0, 'X3':0, 'Y1':0, 'Y2':0, 'Y3':0},
                    {'X1':0, 'X2':0, 'X3':0, 'Y1':1, 'Y2':1, 'Y3':1},
                    {'X1':1, 'X2':1, 'X3':1},
                    {'X1':1, 'X2':1, 'X3':1, 'Y1':0, 'Y2':0, 'Y3':0},
                    {'X1':1, 'X2':1, 'X3':1, 'Y1':1, 'Y2':1, 'Y3':1}]
        sd_edges = [[{'Y1':1}],
                    [{'X1':0, 'X2':0, 'X3':0, 'Y1':1}],
                    [{'X1':1}],
                    [{'X1':1, 'Y1':0, 'Y2':0, 'Y3':0}],
                    [{'X1':1, 'Y1':1, 'Y2':1, 'Y3':1}],
                    [{'X1':1, 'X2':1, 'X3':1, 'Y1':1}]]
        expected_result = (
            False,
            [['001001', '001010', '001011',
              '010001', '010010', '010011',
              '011001', '011010', '011011'],
             ['001000', '010000', '011000'],
             ['001111', '010111', '011111'],
             ['000001', '000010', '000011'],
             ['000000'],
             ['000111'],
             ['111001', '111010', '111011'],
             ['111000'],
             ['111111'],
             ['001100', '001101', '001110',
              '010100', '010101', '010110',
              '011100', '011101', '011110',
              '100100', '100101', '100110',
              '101100', '101101', '101110',
              '110100', '110101', '110110'],
             ['000100', '000101', '000110'],
             ['100001', '100010', '100011', '100100', '100101', '100110',
              '101001', '101010', '101011', '101100', '101101', '101110',
              '110001', '110010', '110011', '110100', '110101', '110110'],
             ['100000', '101000', '110000'],
             ['100111', '101111', '110111'],
             ['111100', '111101', '111110']],
            {'100100': [[{'Y1': 1}], [{'X1': 1}]],
             '100101': [[{'Y1': 1}], [{'X1': 1}]],
             '100110': [[{'Y1': 1}], [{'X1': 1}]],
             '101100': [[{'Y1': 1}], [{'X1': 1}]],
             '101101': [[{'Y1': 1}], [{'X1': 1}]],
             '101110': [[{'Y1': 1}], [{'X1': 1}]],
             '110100': [[{'Y1': 1}], [{'X1': 1}]],
             '110101': [[{'Y1': 1}], [{'X1': 1}]],
             '110110': [[{'Y1': 1}], [{'X1': 1}]]})
        self.assertEqual(get_sd_group_states(nodes, sd_nodes, sd_edges), expected_result)

    def test_percolation_combined(self):
        bnet = """
        X1, X1 | X2 | X3
        X2, X1
        X3, X2
        Y1, Y1 | Y2 | Y3
        Y2, Y1
        Y3, Y2
        """
        nodes, sd_nodes, sd_edges = get_sd_nodes_and_edges(bnet)
        expected_result = (
            False,
            [['001001', '001010', '001011',
              '010001', '010010', '010011',
              '011001', '011010', '011011'],
             ['001000', '010000', '011000'],
             ['001111', '010111', '011111'],
             ['000001', '000010', '000011'],
             ['000000'],
             ['000111'],
             ['111001', '111010', '111011'],
             ['111000'],
             ['111111'],
             ['001100', '001101', '001110',
              '010100', '010101', '010110',
              '011100', '011101', '011110',
              '100100', '100101', '100110',
              '101100', '101101', '101110',
              '110100', '110101', '110110'],
             ['000100', '000101', '000110'],
             ['100001', '100010', '100011', '100100', '100101', '100110',
              '101001', '101010', '101011', '101100', '101101', '101110',
              '110001', '110010', '110011', '110100', '110101', '110110'],
             ['100000', '101000', '110000'],
             ['100111', '101111', '110111'],
             ['111100', '111101', '111110']],
            {'100100': [[{'Y1': 1}], [{'X1': 1}]],
             '100101': [[{'Y1': 1}], [{'X1': 1}]],
             '100110': [[{'Y1': 1}], [{'X1': 1}]],
             '101100': [[{'Y1': 1}], [{'X1': 1}]],
             '101101': [[{'Y1': 1}], [{'X1': 1}]],
             '101110': [[{'Y1': 1}], [{'X1': 1}]],
             '110100': [[{'Y1': 1}], [{'X1': 1}]],
             '110101': [[{'Y1': 1}], [{'X1': 1}]],
             '110110': [[{'Y1': 1}], [{'X1': 1}]]})
        self.assertEqual(get_sd_group_states(nodes, sd_nodes, sd_edges), expected_result)

    def test_empty_list_of_nodes(self):
        nodes = []
        sd_nodes = [{}, {'B': 0}, {'A': 0, 'B': 0}]
        sd_edges = [[], [{'B': 0}], [{'A': 0, 'B': 0}]]
        with self.assertRaises(ValueError):
            get_sd_group_states(nodes, sd_nodes, sd_edges, DEBUG=True)

    def test_empty_list_of_sd_nodes(self):
        nodes = ['A', 'B', 'C']
        sd_nodes = []
        sd_edges = [[], [{'B': 0}], [{'A': 0, 'B': 0}]]
        with self.assertRaises(ValueError):
            get_sd_group_states(nodes, sd_nodes, sd_edges, DEBUG=True)
    
    def test_empty_list_of_SD_edges(self):
        nodes = ['A', 'B']
        sd_nodes = [{}]
        sd_edges = []
        expected_result = (True, [['00', '01', '10', '11']], {})
        self.assertEqual(get_sd_group_states(nodes, sd_nodes, sd_edges), expected_result)

    def test_single_node(self):
        nodes = ['A']
        sd_nodes = [{}, {'A': 0}]
        sd_edges = [[]]
        expected_result = (True, [['1'], ['0'], []], {})
        self.assertEqual(get_sd_group_states(nodes, sd_nodes, sd_edges), expected_result)

    def test_duplicate_SD_nodes(self):
        nodes = ['A', 'B', 'C']
        sd_nodes = [{}, {}, {'A': 0, 'B': 0}]
        sd_edges = [[], [], [{'A': 0, 'B': 0}]]
        with self.assertRaises(ValueError):
            get_sd_group_states(nodes, sd_nodes, sd_edges, DEBUG=True)


class TestStatesToindices(unittest.TestCase):
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
        expected_output = [[1, 2, 6], [0, 7]]
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


if __name__ == '__main__':
    unittest.main()