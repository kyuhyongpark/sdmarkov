import unittest

from sdmarkov.succession_diagram import (
    get_sd_nodes_and_edges,
    sort_sd_nodes,
    trap_spaces_overlap,
    intersect_trap_spaces,
    close_trap_spaces_under_intersection,
    validate_no_missing_intersections,
    build_sd_trap_spaces,
    generate_states,
    get_binary_states,
    assign_states_to_trap_spaces,
)


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
                           [{'A': 1},
                            {'A': 1, 'B': 0}])

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
                           [{'B': 1},
                            {'A': 0, 'B': 0},
                            {'A': 1},])

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
                           [{'A': 0, 'B': 0},
                            {'A': 0, 'B': 1},
                            {'A': 1, 'B': 0},
                            {'A': 1, 'B': 1}])
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
                           [{'B': 0},
                            {'A': 0},
                            {'A': 0, 'B': 0, 'C': 0},
                            {'A': 1, 'B': 0, 'D': 1},
                            {'A': 1, 'B': 1, 'D': 1},])

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
                           [{'Y1':1},
                            {'X1':0, 'X2':0, 'X3':0, 'Y1':1},
                            {'X1':1},
                            {'X1':1, 'Y1':0, 'Y2':0, 'Y3':0},
                            {'X1':1, 'Y1':1, 'Y2':1, 'Y3':1},
                            {'X1':1, 'X2':1, 'X3':1, 'Y1':1}])

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


class TestTrapSpacesOverlap(unittest.TestCase):
    def test_trap_spaces_overlap_basic(self):
        assert trap_spaces_overlap({"A": 1}, {"A": 1}) is True
        assert trap_spaces_overlap({"A": 1}, {"A": 0}) is False


    def test_trap_spaces_overlap_partial(self):
        assert trap_spaces_overlap({"A": 1}, {"B": 0}) is True
        assert trap_spaces_overlap({"A": 1}, {"A": 1, "B": 0}) is True
        assert trap_spaces_overlap({"A": 1}, {"A": 0, "B": 0}) is False


    def test_trap_spaces_overlap_empty(self):
        assert trap_spaces_overlap({}, {}) is True
        assert trap_spaces_overlap({}, {"A": 1}) is True
        assert trap_spaces_overlap({"A": 1}, {}) is True


    def test_trap_spaces_overlap_symmetric(self):
        a = {"A": 1, "B": 0}
        b = {"B": 0, "C": 1}
        assert trap_spaces_overlap(a, b) is True
        assert trap_spaces_overlap(b, a) is True


class TestIntersectTrapSpaces(unittest.TestCase):
    def test_intersect_basic(self):
        assert intersect_trap_spaces({"A": 1}, {"A": 1}) == {"A": 1}
        assert intersect_trap_spaces({"A": 1}, {"A": 0}) is None


    def test_intersect_partial(self):
        assert intersect_trap_spaces({"A": 1}, {"B": 0}) == {"A": 1, "B": 0}
        assert intersect_trap_spaces({"A": 1}, {"A": 1, "B": 0}) == {"A": 1, "B": 0}
        assert intersect_trap_spaces({"A": 1}, {"A": 0, "B": 0}) is None


    def test_intersect_empty(self):
        assert intersect_trap_spaces({}, {}) == {}
        assert intersect_trap_spaces({}, {"A": 1}) == {"A": 1}
        assert intersect_trap_spaces({"A": 1}, {}) == {"A": 1}


    def test_intersect_symmetric(self):
        a = {"A": 1, "B": 0}
        b = {"B": 0, "C": 1}

        assert intersect_trap_spaces(a, b) == {"A": 1, "B": 0, "C": 1}
        assert intersect_trap_spaces(b, a) == {"A": 1, "B": 0, "C": 1}


    def test_intersect_containment(self):
        a = {"A": 1}
        b = {"A": 1, "B": 0}

        assert intersect_trap_spaces(a, b) == b
        assert intersect_trap_spaces(b, a) == b


class TestCloseTrapSpacesUnderIntersection(unittest.TestCase):
    def test_no_overlap(self):
        trap_spaces = [
            {"A": 1},
            {"A": 0},
        ]
        closed = close_trap_spaces_under_intersection(trap_spaces)
        assert len(closed) == 2
        assert {"A": 1} in closed
        assert {"A": 0} in closed


    def test_simple_intersection(self):
        trap_spaces = [
            {"A": 1},
            {"B": 0},
        ]
        closed = close_trap_spaces_under_intersection(trap_spaces)
        assert {"A": 1} in closed
        assert {"B": 0} in closed
        assert {"A": 1, "B": 0} in closed
        assert len(closed) == 3


    def test_chain_intersections(self):
        trap_spaces = [
            {"A": 1},
            {"B": 0},
            {"C": 1},
        ]
        closed = close_trap_spaces_under_intersection(trap_spaces)

        expected = [
            {"A": 1},
            {"B": 0},
            {"C": 1},
            {"A": 1, "B": 0},
            {"A": 1, "C": 1},
            {"B": 0, "C": 1},
            {"A": 1, "B": 0, "C": 1},
        ]

        for ts in expected:
            assert ts in closed

        assert len(closed) == len(expected)


    def test_containment_only(self):
        trap_spaces = [
            {"A": 1},
            {"A": 1, "B": 0},
        ]
        closed = close_trap_spaces_under_intersection(trap_spaces)

        assert {"A": 1} in closed
        assert {"A": 1, "B": 0} in closed
        assert len(closed) == 2


    def test_empty_trap_space(self):
        trap_spaces = [
            {},
            {"A": 1},
            {"B": 0},
        ]
        closed = close_trap_spaces_under_intersection(trap_spaces)

        expected = [
            {},
            {"A": 1},
            {"B": 0},
            {"A": 1, "B": 0},
        ]

        for ts in expected:
            assert ts in closed

        assert len(closed) == len(expected)


class TestValidateNoMissingIntersections(unittest.TestCase):
    def test_valid_simple(self):
        trap_spaces = [
            {"A": 1},
            {"B": 0},
            {"A": 1, "B": 0},
        ]
        assert validate_no_missing_intersections(trap_spaces) is True


    def test_missing_intersection(self):
        trap_spaces = [
            {"A": 1},
            {"B": 0},
        ]
        assert validate_no_missing_intersections(trap_spaces) is False


    def test_no_overlap(self):
        trap_spaces = [
            {"A": 1},
            {"A": 0},
        ]
        assert validate_no_missing_intersections(trap_spaces) is True


    def test_containment_only(self):
        trap_spaces = [
            {"A": 1},
            {"A": 1, "B": 0},
        ]
        assert validate_no_missing_intersections(trap_spaces) is True


    def test_chain_intersections_valid(self):
        trap_spaces = [
            {"A": 1},
            {"B": 0},
            {"C": 1},
            {"A": 1, "B": 0},
            {"A": 1, "C": 1},
            {"B": 0, "C": 1},
            {"A": 1, "B": 0, "C": 1},
        ]
        assert validate_no_missing_intersections(trap_spaces) is True


    def test_empty_trap_space(self):
        trap_spaces = [
            {},
            {"A": 1},
            {"B": 0},
            {"A": 1, "B": 0},
        ]
        assert validate_no_missing_intersections(trap_spaces) is True


class TestBuildSDTrapSpaces(unittest.TestCase):
    def test_returns_nodes_and_trap_spaces(self):
        bnet = "A, A\nB, B"
        nodes, trap_spaces = build_sd_trap_spaces(bnet)

        assert isinstance(nodes, list)
        assert isinstance(trap_spaces, list)

        for ts in trap_spaces:
            assert isinstance(ts, dict)

    def test_result_closed_under_intersection(self):
        bnet = "A, A | B\nB, B"
        _, trap_spaces = build_sd_trap_spaces(bnet)

        # Must always be closed under intersection
        assert validate_no_missing_intersections(trap_spaces)

    def test_no_duplicate_trap_spaces(self):
        bnet = "A, A | B\nB, B"
        _, trap_spaces = build_sd_trap_spaces(bnet)

        keys = [frozenset(ts.items()) for ts in trap_spaces]
        assert len(keys) == len(set(keys))


    def test_trivial_network(self):
        bnet = "A, A"
        nodes, trap_spaces = build_sd_trap_spaces(bnet)

        assert nodes == ["A"]
        assert {"A": 0} in trap_spaces and {"A": 1} in trap_spaces

    def test_multi_motif_edge(self):
        bnet="""
        A, A | B
        B, A | B
        C, A & B 
        """
        nodes, trap_spaces = build_sd_trap_spaces(bnet)
        assert nodes == ["A", "B", "C"]
        expected = [
            {},
            {"B": 1},
            {"A": 0, "B": 0},
            {"A": 0, "B": 0, "C": 0},
            {"A": 1},
            {"A": 1, "B": 1},
            {"A": 1, "B": 1, "C": 1},
        ]
        self.assertEqual(trap_spaces, expected)

    def test_example(self):
        bnet="""
        A, A | B & C
        B, B & !C
        C, B & !C | !C & !D | !B & C & D
        D, !A & !B & !C & !D | !A & C & D
        """
        nodes, trap_spaces = build_sd_trap_spaces(bnet)
        expected_result = [
            {},
            {'B': 0},
            {'A': 0, 'B': 0},
            {'A': 0, 'B': 0, 'C': 1, 'D': 1},
            {'A': 1},
            {'A': 1, 'D': 0},
            {'A': 1, 'B': 0},
            {'A': 1, 'B': 0, 'D': 0},]
        self.assertEqual(trap_spaces, expected_result)

    def test_intersection(self):
        bnet="""
        A, A & D
        B, A & B
        C, B
        D, A
        """
        nodes, trap_spaces = build_sd_trap_spaces(bnet)
        expected_result = [
            {},
            {'B': 0},
            {'B': 0, 'C': 0},
            {'A': 0},
            {'A': 0, 'B': 0},
            {'A': 0, 'B': 0, 'C': 0},
            {'A': 0, 'B': 0, 'C': 0, 'D': 0},
            {'A': 1, 'D': 1},
            {'A': 1, 'B': 0, 'D': 1},
            {'A': 1, 'B': 0, 'C': 0, 'D': 1},
            {'A': 1, 'B': 1, 'D': 1},
            {'A': 1, 'B': 1, 'C': 1, 'D': 1},
            ]
        self.assertEqual(trap_spaces, expected_result)

    def test_percolation(self):
        bnet = """
        X1, X1 | X2 | X3
        X2, X1
        X3, X2
        Y1, Y1 | Y2 | Y3
        Y2, Y1
        Y3, Y2
        """
        nodes, trap_spaces = build_sd_trap_spaces(bnet)
        expected_result = [
            {},
            {'Y1':0, 'Y2':0, 'Y3':0},
            {'Y1':1},
            {'Y1':1, 'Y2':1, 'Y3':1},
            {'X1':0, 'X2':0, 'X3':0},
            {'X1':0, 'X2':0, 'X3':0, 'Y1':0, 'Y2':0, 'Y3':0},
            {'X1':0, 'X2':0, 'X3':0, 'Y1':1},
            {'X1':0, 'X2':0, 'X3':0, 'Y1':1, 'Y2':1, 'Y3':1},
            {'X1':1},
            {'X1':1, 'Y1':0, 'Y2':0, 'Y3':0},
            {'X1':1, 'Y1':1},
            {'X1':1, 'Y1':1, 'Y2':1, 'Y3':1},
            {'X1':1, 'X2':1, 'X3':1},
            {'X1':1, 'X2':1, 'X3':1, 'Y1':0, 'Y2':0, 'Y3':0},
            {'X1':1, 'X2':1, 'X3':1, 'Y1':1},
            {'X1':1, 'X2':1, 'X3':1, 'Y1':1, 'Y2':1, 'Y3':1},]
        self.assertEqual(trap_spaces, expected_result)

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


class TestAssignStatesToTrapSpaces(unittest.TestCase):
    def test_single_trap_space_all_states(self):
        nodes = ["A", "B"]
        trap_spaces = [{}]

        states = assign_states_to_trap_spaces(nodes, trap_spaces, DEBUG=True)

        self.assertEqual(
            states,
            [["00", "01", "10", "11"]],
        )

    def test_disjoint_trap_spaces(self):
        nodes = ["A", "B"]
        trap_spaces = [
            {"A": 0},
            {"A": 1},
        ]

        states = assign_states_to_trap_spaces(nodes, trap_spaces, DEBUG=True)

        self.assertEqual(
            states,
            [["00", "01"], ["10", "11"]],
        )

    def test_multiple_variables(self):
        nodes = ["A", "B", "C"]
        trap_spaces = [
            {"A": 0, "B": 0},
            {"A": 0, "B": 1},
            {"A": 1},
        ]

        states = assign_states_to_trap_spaces(nodes, trap_spaces, DEBUG=True)

        all_states = [s for g in states for s in g]
        self.assertEqual(len(all_states), 8)
        self.assertEqual(len(set(all_states)), 8)


class TestEndToEnd(unittest.TestCase):
    def test_example_from_docstring(self):
        bnet="""
        A, A | B & C
        B, B & !C
        C, B & !C | !C & !D | !B & C & D
        D, !A & !B & !C & !D | !A & C & D
        """
        nodes, trap_spaces = build_sd_trap_spaces(bnet)
        states = assign_states_to_trap_spaces(nodes, trap_spaces)
        expected_result = [
            ['0100', '0101', '0110', '0111'],
            [],
            ['0000', '0001', '0010'],
            ['0011'],
            ['1101', '1111'],
            ['1100', '1110'],
            ['1001', '1011'],
            ['1000', '1010'],]
        self.assertEqual(states, expected_result)

    def test_minimal(self):
        bnet="""
        A, A | B & C
        B, B & !C
        C, B & !C | !C & !D | !B & C & D
        D, !A & !B & !C & !D | !A & C & D
        """
        nodes, trap_spaces = build_sd_trap_spaces(bnet, minimal=True)
        states = assign_states_to_trap_spaces(nodes, trap_spaces)
        expected_result = [
            ['0011'],
            ['1000', '1010']]
        self.assertEqual(states, expected_result)

    def test_multi_motif_edge(self):
        bnet="""
        A, A | B
        B, A | B
        C, A & B 
        """
        nodes, trap_spaces = build_sd_trap_spaces(bnet)
        states = assign_states_to_trap_spaces(nodes, trap_spaces)
        expected_result = [
            [],
            ["010", "011"],
            ["001"],
            ["000"],
            ["100", "101"],
            ["110"],
            ["111"],
        ]
        self.assertEqual(states, expected_result)

    def test_intersection(self):
        bnet="""
        A, A & D
        B, A & B
        C, B
        D, A
        """
        nodes, trap_spaces = build_sd_trap_spaces(bnet)
        states = assign_states_to_trap_spaces(nodes, trap_spaces)
        expected_result = [
            ['1100', '1110',],
            ['1010'],
            ['1000'],
            ['0100', '0101', '0110', '0111',],
            ["0010", "0011"],
            ['0001'],
            ['0000'],
            [],
            ['1011'],
            ['1001'],
            ['1101'],
            ['1111'],
            ]

        self.assertEqual(states, expected_result)

    def test_percolation(self):
        bnet = """
        X1, X1 | X2 | X3
        X2, X1
        X3, X2
        Y1, Y1 | Y2 | Y3
        Y2, Y1
        Y3, Y2
        """

        nodes, trap_spaces = build_sd_trap_spaces(bnet)
        states = assign_states_to_trap_spaces(nodes, trap_spaces)

        expected_result = [
            ['001001', '001010', '001011', '010001', '010010', '010011', '011001', '011010', '011011'],
            ['001000', '010000', '011000'],
            ['001100', '001101', '001110', '010100', '010101', '010110', '011100', '011101', '011110'],
            ['001111', '010111', '011111'],
            ['000001', '000010', '000011'],
            ['000000'],
            ['000100', '000101', '000110'],
            ['000111'],
            ['100001', '100010', '100011', '101001', '101010', '101011', '110001', '110010', '110011'],
            ['100000', '101000', '110000'],
            ['100100', '100101', '100110', '101100', '101101', '101110', '110100', '110101', '110110'],
            ['100111', '101111', '110111'],
            ['111001', '111010', '111011'],
            ['111000'],
            ['111100', '111101', '111110'],
            ['111111'],]

        self.assertEqual(states, expected_result)


if __name__ == '__main__':
    unittest.main()