import unittest

from sdmarkov.grouping import sd_grouping, null_grouping
from sdmarkov.grouping import divide_list_into_sublists, random_grouping


class TestSdGrouping(unittest.TestCase):
    def test_valid_bnet(self):
        bnet = """
        A, A | B & C
        B, B & !C
        C, B & !C | !C & !D | !B & C & D
        D, !A & !B & !C & !D | !A & C & D
        """
        expected_output = [[4, 5, 6, 7], [0, 1, 2], [3], [12, 14], [8, 10], [13, 15], [9, 11]]
        output = sd_grouping(bnet)
        self.assertEqual(output, expected_output)

    def test_scrambled_bnet(self):
        bnet = """
        B, B & !C
        D, !A & !B & !C & !D | !A & C & D
        A, A | B & C
        C, B & !C | !C & !D | !B & C & D
        """
        expected_output = [[4, 5, 6, 7], [0, 1, 2], [3], [12, 14], [8, 10], [13, 15], [9, 11]]

        output = sd_grouping(bnet)
        self.assertEqual(output, expected_output)

    def test_duplicate_states(self):
        bnet = """
        A, A & D
        B, A & B
        C, B
        D, A
        """
        expected_output = [[12, 14,],
                           [8],
                           [0],
                           [9],
                           [15],
                           [10],
                           [4, 5, 6, 7,],
                           [1,],
                           [11,],
                           [13,],
                           [2, 3,],]
        self.assertEqual(sd_grouping(bnet, DEBUG=True), expected_output)

    def test_triple(self):
        bnet="""
        A, A | a
        a, A
        B, B | b
        b, B
        C, C | c
        c, C
        """
        print(sd_grouping(bnet, DEBUG=True))

    def test_valid_bnet_debug(self):
        bnet = """
        A, A | B & C
        B, B & !C
        C, B & !C | !C & !D | !B & C & D
        D, !A & !B & !C & !D | !A & C & D
        """
        expected_output = [[4, 5, 6, 7], [0, 1, 2], [3], [12, 14], [8, 10], [13, 15], [9, 11]]
        output = sd_grouping(bnet, DEBUG=True)
        self.assertEqual(output, expected_output)


class TestNullGrouping(unittest.TestCase):
    def test_valid_bnet(self):
        bnet = """
        A, A | B & C
        B, B & !C
        C, B & !C | !C & !D | !B & C & D
        D, !A & !B & !C & !D | !A & C & D
        """
        expected_output = [[0, 1, 2, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15], [3], [8, 10]]
        output = null_grouping(bnet)
        self.assertEqual(output, expected_output)

    def test_scrambled_bnet(self):
        bnet = """
        B, B & !C
        D, !A & !B & !C & !D | !A & C & D
        A, A | B & C
        C, B & !C | !C & !D | !B & C & D
        """
        expected_output = [[0, 1, 2, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15], [3], [8, 10]]
        output = null_grouping(bnet)
        self.assertEqual(output, expected_output)

    def test_valid_bnet_debug(self):
        bnet = """
        A, A | B & C
        B, B & !C
        C, B & !C | !C & !D | !B & C & D
        D, !A & !B & !C & !D | !A & C & D
        """
        expected_output = [[0, 1, 2, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15], [3], [8, 10]]
        output = null_grouping(bnet, DEBUG=True)
        self.assertEqual(output, expected_output)


class TestDivideListIntoSublists(unittest.TestCase):
    def test_simple_case(self):
        lst = [1, 2, 3, 4, 5]
        N = 2
        m = 2
        result = divide_list_into_sublists(lst, N, m)
        self.assertEqual(len(result), N)
        self.assertTrue(all(len(sublist) >= m for sublist in result))

    def test_not_enough_elements(self):
        lst = [1, 2, 3]
        N = 2
        m = 3
        with self.assertRaises(ValueError):
            divide_list_into_sublists(lst, N, m)

    def test_N_equals_1(self):
        lst = [1, 2, 3]
        N = 1
        m = 1
        result = divide_list_into_sublists(lst, N, m)
        self.assertEqual(result, [lst])

    def test_N_equals_len_lst(self):
        lst = [1, 2, 3]
        N = len(lst)
        m = 1
        result = divide_list_into_sublists(lst, N, m)
        self.assertEqual(result, [[1], [2], [3]])

    def test_DEBUG_invalid_N(self):
        lst = [1, 2, 3]
        N = 4
        m = 1
        with self.assertRaises(ValueError):
            divide_list_into_sublists(lst, N, m)

    def test_DEBUG_invalid_m(self):
        lst = [1, 2, 3]
        N = 2
        m = 0
        with self.assertRaises(ValueError):
            divide_list_into_sublists(lst, N, m)

    def test_DEBUG_N_m_greater_than_len_lst(self):
        lst = [1, 2, 3]
        N = 2
        m = 3
        with self.assertRaises(ValueError):
            divide_list_into_sublists(lst, N, m)


class TestRandomGrouping(unittest.TestCase):
    def test_valid_input(self):
        sd_indices = [[4, 5, 6, 7], [0, 1, 2], [3], [13, 15], [12, 14], [9, 11], [8, 10]]
        null_indices = [[0, 1, 2, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15], [3], [8, 10]]
        result = random_grouping(sd_indices, null_indices)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 7)
        self.assertTrue(all(len(sublist) >= 1 for sublist in result))
        self.assertTrue([3] in result)
        self.assertTrue([8, 10] in result)

    def test_debug_true(self):
        sd_indices = [[4, 5, 6, 7], [0, 1, 2], [3], [13, 15], [12, 14], [9, 11], [8, 10]]
        null_indices = [[0, 1, 2, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15], [3], [8, 10]]
        result = random_grouping(sd_indices, null_indices, DEBUG=True)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 7)
        self.assertTrue(all(len(sublist) >= 1 for sublist in result))
        self.assertTrue([3] in result)
        self.assertTrue([8, 10] in result)

    def test_invalid_input(self):
        sd_indices = [[5, 6, 7], [0, 1, 2], [3, 4], [13, 15], [12, 14], [9, 11], [8, 10]]
        null_indices = [[0, 1, 2, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15], [3], [8, 10]]
        with self.assertRaises(ValueError):
            random_grouping(sd_indices, null_indices, DEBUG=True)

    def test_smallest_group_size_too_large(self):
        sd_indices = [[4, 5, 6, 7], [0, 1, 2], [3], [13, 15], [12, 14], [9, 11], [8, 10]]
        null_indices = [[0, 1, 2, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15], [3], [8, 10]]
        with self.assertRaises(ValueError):
            random_grouping(sd_indices, null_indices, smallest_group_size=4)

    def test_empty_transient(self):
        sd_indices = [[0], [1, 2, 3]]
        null_indices = [[0], [1, 2, 3]]
        expected_result = [[0], [1, 2, 3]]
        result = random_grouping(sd_indices, null_indices)
        self.assertEqual(result, expected_result)

    def test_seed_value(self):
        sd_indices = [[4, 5, 6, 7], [0, 1, 2], [3], [13, 15], [12, 14], [9, 11], [8, 10]]
        null_indices = [[0, 1, 2, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15], [3], [8, 10]]
        result1 = random_grouping(sd_indices, null_indices, seed=42)
        result2 = random_grouping(sd_indices, null_indices, seed=42)
        self.assertEqual(result1, result2)


if __name__ == "__main__":
    unittest.main()

    