import unittest

from helper import states_to_indices, indices_to_states


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


if __name__ == '__main__':
    unittest.main()