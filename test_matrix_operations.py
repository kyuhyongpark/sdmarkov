import unittest

import numpy as np

from matrix_operations import reorder_matrix
from matrix_operations import compress_matrix
from matrix_operations import expand_matrix
from matrix_operations import get_rms_diff
from matrix_operations import get_dkl
from matrix_operations import get_reachability

class TestReorderMatrix(unittest.TestCase):
    def test_reorder_matrix_3x3_valid(self):
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        index_list = [2, 1, 0]
        expected_output = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
        self.assertTrue(np.array_equal(reorder_matrix(matrix, index_list), expected_output))

    def test_reorder_matrix_4x4_valid(self):
        matrix = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        index_list = [3, 2, 1, 0]
        expected_output = np.array([[16, 15, 14, 13], [12, 11, 10, 9], [8, 7, 6, 5], [4, 3, 2, 1]])
        self.assertTrue(np.array_equal(reorder_matrix(matrix, index_list), expected_output))

    def test_large_example(self):
        matrix = np.array(
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
        index_list = [5,7,4,6,13,15,12,14,9,11,8,10,0,1,2,3]
        expected_output = np.array(
            [[1/2, 1/4, 1/4, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  ],
             [1/4, 1/4, 0,   0,   0,   1/4, 0,   0,   0,   0,   0,   0,   0,   0,   0,   1/4,],
             [0,   0,   3/4, 1/4, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  ],
             [0,   0,   1/4, 1/4, 0,   0,   0,   1/4, 0,   0,   0,   0,   0,   0,   1/4, 0,  ],
             [0,   0,   0,   0,   1/2, 1/4, 1/4, 0,   0,   0,   0,   0,   0,   0,   0,   0,  ],
             [0,   0,   0,   0,   1/4, 1/4, 0,   1/4, 0,   1/4, 0,   0,   0,   0,   0,   0,  ],
             [0,   0,   0,   0,   0,   0,   3/4, 1/4, 0,   0,   0,   0,   0,   0,   0,   0,  ],
             [0,   0,   0,   0,   0,   0,   1/4, 1/2, 0,   0,   0,   1/4, 0,   0,   0,   0,  ],
             [0,   0,   0,   0,   0,   0,   0,   0,   3/4, 0,   1/4, 0,   0,   0,   0,   0,  ],
             [0,   0,   0,   0,   0,   0,   0,   0,   0,   3/4, 0,   1/4, 0,   0,   0,   0,  ],
             [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   3/4, 1/4, 0,   0,   0,   0,  ],
             [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1/4, 3/4, 0,   0,   0,   0,  ],
             [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1/2, 1/4, 1/4, 0,  ],
             [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1/4, 3/4, 0,   0,  ],
             [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1/4, 0,   3/4, 0,  ],
             [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,  ]])
        self.assertTrue(np.array_equal(reorder_matrix(matrix, index_list), expected_output))  

    def test_reorder_matrix_3x3_invalid(self):
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        index_list = [2, 1]  # length mismatch
        with self.assertRaises(ValueError):
            reorder_matrix(matrix, index_list)

    def test_reorder_matrix_non_square(self):
        matrix = np.array([[1, 2, 3], [4, 5, 6]])  # non-square matrix
        index_list = [1, 0]
        with self.assertRaises(ValueError):
            reorder_matrix(matrix, index_list)


class TestCompressMatrix(unittest.TestCase):
    def test_single_index_group(self):
        matrix = np.array([[1, 0, 0], [0, 1/2, 1/2], [0, 0, 1]])
        index_groups = [[0, 1]]
        expected_result = np.array([[3/4, 1/4], [0, 1]])
        self.assertTrue(np.allclose(compress_matrix(matrix, index_groups), expected_result))

    def test_empty_index_group(self):
        matrix = np.array([[1, 0, 0], [0, 1/2, 1/2], [0, 0, 1]])
        index_groups = [[2], [], [0, 1]]
        expected_result = np.array([[1, 0], [1/4, 3/4]])
        self.assertTrue(np.allclose(compress_matrix(matrix, index_groups), expected_result))

    def test_no_compression(self):
        matrix = np.array([[1, 0, 0], [0, 1/2, 1/2], [0, 0, 1]])
        index_groups = [[0], [1], [2]]
        self.assertTrue(np.allclose(compress_matrix(matrix, index_groups), matrix))

    def test_multiple_index_groups(self):
        matrix = np.array([[1, 0, 0], [0, 1/2, 1/2], [0, 0, 1]])
        index_groups = [[0, 1], [2]]
        expected_result = np.array([[3/4, 1/4], [0, 1]])
        self.assertTrue(np.allclose(compress_matrix(matrix, index_groups), expected_result))

    def test_large_matrix(self):
        matrix = np.array(
            [[0.5 , 0.25, 0.25, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,0.  ],
             [0.25, 0.25, 0.  , 0.  , 0.  , 0.25, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,0.25],
             [0.  , 0.  , 0.75, 0.25, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,0.  ],
             [0.  , 0.  , 0.25, 0.25, 0.  , 0.  , 0.  , 0.25, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.25,0.  ],
             [0.  , 0.  , 0.  , 0.  , 0.5 , 0.25, 0.25, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,0.  ],
             [0.  , 0.  , 0.  , 0.  , 0.25, 0.25, 0.  , 0.25, 0.  , 0.25, 0.  , 0.  , 0.  , 0.  , 0.  ,0.  ],
             [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.75, 0.25, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,0.  ],
             [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 0.5 , 0.  , 0.  , 0.  , 0.25, 0.  , 0.  , 0.  ,0.  ],
             [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.75, 0.  , 0.25, 0.  , 0.  , 0.  , 0.  ,0.  ],
             [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.75, 0.  , 0.25, 0.  , 0.  , 0.  ,0.  ],
             [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.75, 0.25, 0.  , 0.  , 0.  ,0.  ],
             [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 0.75, 0.  , 0.  , 0.  ,0.  ],
             [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.5 , 0.25, 0.25,0.  ],
             [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 0.75, 0.  ,0.  ],
             [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 0.  , 0.75,0.  ],
             [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ,1.  ]])
        index_groups = [[0,1,2,3,4,5], [6,7],[8,9],[10,11],[12,13,14],[15]]
        expected_result = np.array(
            [[0.75      , 0.125     , 0.04166667, 0.        , 0.04166667, 0.04166667],
             [0.        , 0.875     , 0.        , 0.125     , 0.        , 0.        ],
             [0.        , 0.        , 0.75      , 0.25      , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 1.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 1.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        , 1.        ]])
        self.assertTrue(np.allclose(compress_matrix(matrix, index_groups), expected_result))

    def test_all_rows_and_columns(self):
        matrix = np.array([[1, 0, 0], [0, 1/2, 1/2], [0, 0, 1]])
        index_groups = [[0, 1, 2]]
        expected_result = np.array([[1]])
        self.assertTrue(np.allclose(compress_matrix(matrix, index_groups), expected_result))

    def test_empty_index_group_list(self):
        matrix = np.array([[1, 0, 0], [0, 1/2, 1/2], [0, 0, 1]])
        index_groups = []
        expected_result = matrix
        self.assertTrue(np.allclose(compress_matrix(matrix, index_groups), expected_result))

    def test_non_numeric_matrix(self):
        matrix = np.array([['a', 'b', 'c'], ['d', 'e', 'f'], ['g', 'h', 'i']])
        index_groups = [[0, 1]]
        with self.assertRaises(TypeError):
            compress_matrix(matrix, index_groups)

    def test_index_groups_out_of_bounds(self):
        matrix = np.array([[1, 0, 0], [0, 1/2, 1/2], [0, 0, 1]])
        index_groups = [[0, 1], [3, 4]]
        with self.assertRaises(ValueError):
            compress_matrix(matrix, index_groups, DEBUG=True)

    def test_overlapping_index_groups(self):
        matrix = np.array([[1, 0, 0], [0, 1/2, 1/2], [0, 0, 1]])
        index_groups = [[0, 1], [1, 2]]
        with self.assertRaises(ValueError):
            compress_matrix(matrix, index_groups, DEBUG=True)
            
    def test_single_row_matrix(self):
        matrix = np.array([[1, 0, 0]])
        index_groups = [[0, 1]]
        with self.assertRaises(ValueError):
            compress_matrix(matrix, index_groups, DEBUG=True)

    def test_single_column_matrix(self):
        matrix = np.array([[1], [1], [1]])
        index_groups = [[0, 1]]
        with self.assertRaises(ValueError):
            compress_matrix(matrix, index_groups, DEBUG=True)


class TestExpandMatrix(unittest.TestCase):
    def test_example(self):
        matrix = np.array([[3/4, 1/4], [0, 1]])
        index_groups = [[0, 1], [2]]
        expected_result = np.array([[3/8, 3/8, 1/4], [3/8, 3/8, 1/4], [0, 0, 1]])
        self.assertTrue(np.allclose(expand_matrix(matrix, index_groups), expected_result))

    def test_empty_index_group(self):
        matrix = np.array([[3/4, 1/4], [0, 1]])
        index_groups = [[0, 1], [], [2]]
        expected_result = np.array([[3/8, 3/8, 1/4], [3/8, 3/8, 1/4], [0, 0, 1]])
        self.assertTrue(np.allclose(expand_matrix(matrix, index_groups), expected_result))

    def test_large_matrix(self):
        matrix = np.array(
            [[0.75      , 0.125     , 0.04166667, 0.        , 0.04166667, 0.04166667],
             [0.        , 0.875     , 0.        , 0.125     , 0.        , 0.        ],
             [0.        , 0.        , 0.75      , 0.25      , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 1.        , 0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 1.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        , 1.        ]])
        index_groups = [[0,1,2,3,4,5], [6,7],[8,9],[10,11],[12,13,14],[15]]
        expected_result = np.array(
            [[0.125     , 0.125     , 0.125     , 0.125     , 0.125     , 0.125     , 0.0625    ,0.0625    , 0.02083333, 0.02083333, 0.        , 0.        , 0.01388889, 0.01388889,0.01388889, 0.04166667],
             [0.125     , 0.125     , 0.125     , 0.125     , 0.125     , 0.125     , 0.0625    ,0.0625    , 0.02083333, 0.02083333, 0.        , 0.        , 0.01388889, 0.01388889,0.01388889, 0.04166667],
             [0.125     , 0.125     , 0.125     , 0.125     , 0.125     , 0.125     , 0.0625    ,0.0625    , 0.02083333, 0.02083333, 0.        , 0.        , 0.01388889, 0.01388889,0.01388889, 0.04166667],
             [0.125     , 0.125     , 0.125     , 0.125     , 0.125     , 0.125     , 0.0625    ,0.0625    , 0.02083333, 0.02083333, 0.        , 0.        , 0.01388889, 0.01388889,0.01388889, 0.04166667],
             [0.125     , 0.125     , 0.125     , 0.125     , 0.125     , 0.125     , 0.0625    ,0.0625    , 0.02083333, 0.02083333, 0.        , 0.        , 0.01388889, 0.01388889,0.01388889, 0.04166667],
             [0.125     , 0.125     , 0.125     , 0.125     , 0.125     , 0.125     , 0.0625    ,0.0625    , 0.02083333, 0.02083333, 0.        , 0.        , 0.01388889, 0.01388889,0.01388889, 0.04166667],
             [0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.4375    ,0.4375    , 0.        , 0.        , 0.0625    , 0.0625    , 0.        , 0.        ,0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.4375    ,0.4375    , 0.        , 0.        , 0.0625    , 0.0625    , 0.        , 0.        ,0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,0.        , 0.375     , 0.375     , 0.125     , 0.125     , 0.        , 0.        ,0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,0.        , 0.375     , 0.375     , 0.125     , 0.125     , 0.        , 0.        ,0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,0.        , 0.        , 0.        , 0.5       , 0.5       , 0.        , 0.        ,0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,0.        , 0.        , 0.        , 0.5       , 0.5       , 0.        , 0.        ,0.        , 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,0.        , 0.        , 0.        , 0.        , 0.        , 0.33333333, 0.33333333,0.33333333, 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,0.        , 0.        , 0.        , 0.        , 0.        , 0.33333333, 0.33333333,0.33333333, 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,0.        , 0.        , 0.        , 0.        , 0.        , 0.33333333, 0.33333333,0.33333333, 0.        ],
             [0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,0.        , 1.        ]])
        self.assertTrue(np.allclose(expand_matrix(matrix, index_groups), expected_result))

    def test_too_many_groups(self):
        matrix = np.array([[1/2, 1/2], [1/3, 2/3]])
        index_groups = [[0, 1], [2, 3], [4]]
        with self.assertRaises(ValueError):
            expand_matrix(matrix, index_groups, DEBUG=True)

    def test_overlapping_groups(self):
        matrix = np.array([[1/2, 1/2], [1/3, 2/3]])
        index_groups = [[0, 1], [1, 2]]
        with self.assertRaises(ValueError):
            expand_matrix(matrix, index_groups, DEBUG=True)

    def test_matrix_non_square(self):
        matrix = np.array([[1/2, 1/2], [1/3, 2/3], [1/4, 3/4]])
        index_groups = [[0, 1], [2]]
        with self.assertRaises(ValueError):
            expand_matrix(matrix, index_groups, DEBUG=True)

    def test_matrix_outside_range(self):
        matrix = np.array([[2, 1/2], [1/3, 2/3]])
        index_groups = [[0, 1], [2]]
        with self.assertRaises(ValueError):
            expand_matrix(matrix, index_groups, DEBUG=True)

    def test_matrix_rows_not_summing_to_one(self):
        matrix = np.array([[1/2, 1/2], [1/3, 2/3]])
        matrix[0, 0] = 1/4
        index_groups = [[0, 1], [2]]
        with self.assertRaises(ValueError):
            expand_matrix(matrix, index_groups, DEBUG=True)


class TestGetRmsDiff(unittest.TestCase):

    def test_identical_matrices(self):
        A = np.array([[1, 0], [0, 1]])
        B = np.array([[1, 0], [0, 1]])
        self.assertAlmostEqual(get_rms_diff(A, B), 0)

    def test_different_matrices(self):
        A = np.array([[1, 0], [0, 1]])
        B = np.array([[1, 0], [1/2, 1/2]])
        self.assertAlmostEqual(get_rms_diff(A, B), 0.35355339059327376220042218105242)

    def test_debug_mode_enabled(self):
        A = np.array([[1, 0], [0, 1]])
        B = np.array([[1, 0], [1/2, 1/2]])
        self.assertAlmostEqual(get_rms_diff(A, B, DEBUG=True), 0.35355339059327376220042218105242)

    def test_different_shapes(self):
        A = np.array([[1, 0], [0, 1]])
        B = np.array([[1, 0, 0], [0, 1, 0]])
        with self.assertRaises(ValueError):
            get_rms_diff(A, B, DEBUG=True)

    def test_out_of_range_values(self):
        A = np.array([[1, 0], [0, 1]])
        B = np.array([[1, 0], [0, 2]])
        with self.assertRaises(ValueError):
            get_rms_diff(A, B, DEBUG=True)

    def test_rows_not_summing_to_1(self):
        A = np.array([[1, 0], [0, 1]])
        B = np.array([[1, 0], [0, 2]])
        with self.assertRaises(ValueError):
            get_rms_diff(A, B, DEBUG=True)

class TestGetDKL(unittest.TestCase):

    def test_identical_matrices(self):
        A = np.array([[1, 0], [0, 1]])
        B = np.array([[1, 0], [0, 1]])
        self.assertAlmostEqual(get_dkl(A, B), 0)

    def test_different_matrices(self):
        A = np.array([[1, 0], [0, 1]])
        B = np.array([[1, 0], [1/2, 1/2]])
        self.assertAlmostEqual(get_dkl(A, B), 0.69314718055994530941723212145818)

    def test_debug_mode_enabled(self):
        A = np.array([[1, 0], [0, 1]])
        B = np.array([[1, 0], [1/2, 1/2]])
        self.assertAlmostEqual(get_dkl(A, B, DEBUG=True), 0.69314718055994530941723212145818)

    def test_different_shapes(self):
        A = np.array([[1, 0], [0, 1]])
        B = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        with self.assertRaises(ValueError):
            get_dkl(A, B, DEBUG=True)

    def test_out_of_range_values(self):
        A = np.array([[1, 0], [0, 1]])
        B = np.array([[1, 0], [1/2, 3/2]])
        with self.assertRaises(ValueError):
            get_dkl(A, B, DEBUG=True)

    def test_rows_not_summing_to_1(self):
        A = np.array([[1, 0], [0, 1]])
        B = np.array([[1, 0], [1/2, 1/3]])
        with self.assertRaises(ValueError):
            get_dkl(A, B, DEBUG=True)


class TestGetReachability(unittest.TestCase):
    def test_identical_matrices(self):
        answer = np.array([[1, 0, 1], [0, 1, 0]])
        guess = np.array([[1, 0, 1], [0, 1, 0]])
        TP, FP, TN, FN = get_reachability(answer, guess)
        self.assertEqual(TP, 3)
        self.assertEqual(FP, 0)
        self.assertEqual(TN, 3)
        self.assertEqual(FN, 0)
    def test_guess_all_zeros(self):
        answer = np.array([[1, 0, 1], [0, 1, 0]])
        guess = np.array([[0, 0, 0], [0, 0, 0]])
        TP, FP, TN, FN = get_reachability(answer, guess)
        self.assertEqual(TP, 0)
        self.assertEqual(FP, 0)
        self.assertEqual(TN, 3)
        self.assertEqual(FN, 3)
    def test_answer_all_zeros(self):
        answer = np.array([[0, 0, 0], [0, 0, 0]])
        guess = np.array([[1, 0, 1], [0, 1, 0]])
        TP, FP, TN, FN = get_reachability(answer, guess)
        self.assertEqual(TP, 0)
        self.assertEqual(FP, 3)
        self.assertEqual(TN, 3)
        self.assertEqual(FN, 0)
    def test_guess_all_ones_answer_all_zeros(self):
        answer = np.array([[0, 0, 0], [0, 0, 0]])
        guess = np.array([[1, 1, 1], [1, 1, 1]])
        TP, FP, TN, FN = get_reachability(answer, guess)
        self.assertEqual(TP, 0)
        self.assertEqual(FP, 6)
        self.assertEqual(TN, 0)
        self.assertEqual(FN, 0)
    def test_guess_all_ones_answer_all_ones(self):
        answer = np.array([[1, 1, 1], [1, 1, 1]])
        guess = np.array([[1, 1, 1], [1, 1, 1]])
        TP, FP, TN, FN = get_reachability(answer, guess)
        self.assertEqual(TP, 6)
        self.assertEqual(FP, 0)
        self.assertEqual(TN, 0)
        self.assertEqual(FN, 0)
    def test_random_matrices(self):
        np.random.seed(0)
        answer = np.random.randint(0, 2, size=(10, 10))
        guess = np.random.randint(0, 2, size=(10, 10))
        TP, FP, TN, FN = get_reachability(answer, guess)
        # Check that the sum of TP, FP, TN, and FN is equal to the total number of elements
        self.assertEqual(TP + FP + TN + FN, 100)

if __name__ == '__main__':
    unittest.main()