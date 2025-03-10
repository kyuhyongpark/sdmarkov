import unittest

import numpy as np

from matrix_operations import reorder_matrix
from matrix_operations import compress_matrix
from matrix_operations import expand_matrix
from matrix_operations import get_rms_diff
from matrix_operations import get_dkl
from matrix_operations import get_reachability
from matrix_operations import is_block_triangular
from matrix_operations import get_block_triangular
from matrix_operations import enforce_asynchronous
from transition_matrix import get_stg
from scc_dags import get_scc_dag

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

    def test_identical_matrices(self):
        answer = np.array([[1, 0], [0, 1]])
        guess = np.array([[1, 0], [0, 1]])
        TP, FP, TN, FN = get_reachability(answer, guess)
        self.assertEqual(TP, 2)
        self.assertEqual(FP, 0)
        self.assertEqual(TN, 2)
        self.assertEqual(FN, 0)

    def test_different_matrices(self):
        answer = np.array([[1, 0], [0, 1]])
        guess = np.array([[1, 1], [0, 0]])
        TP, FP, TN, FN = get_reachability(answer, guess)
        self.assertEqual(TP, 1)
        self.assertEqual(FP, 1)
        self.assertEqual(TN, 1)
        self.assertEqual(FN, 1)

    def test_matrices_with_different_shapes(self):
        answer = np.array([[1, 0], [0, 1]])
        guess = np.array([[1, 0, 1], [0, 1, 0]])
        with self.assertRaises(ValueError):
            get_reachability(answer, guess)

    def test_type_parameter_with_invalid_value(self):
        answer = np.array([[1, 0], [0, 1]])
        guess = np.array([[1, 0], [0, 1]])
        with self.assertRaises(ValueError):
            get_reachability(answer, guess, get_type="invalid")

    def test_type_parameter_with_attractor_and_missing_scc_indices_or_attractor_states(self):
        answer = np.array([[1, 0], [0, 1]])
        guess = np.array([[1, 0], [0, 1]])
        with self.assertRaises(ValueError):
            get_reachability(answer, guess, get_type="attractor")

    def test_type_parameter_with_basin_and_missing_scc_indices_or_attractor_states(self):
        answer = np.array([[1, 0], [0, 1]])
        guess = np.array([[1, 0], [0, 1]])
        with self.assertRaises(ValueError):
            get_reachability(answer, guess, get_type="basin")

    def test_type_parameter_with_attractor_and_correct_scc_indices_and_attractor_states(self):
        answer = np.array([[1, 0], [0, 1]])
        guess = np.array([[1, 0], [0, 1]])
        scc_indices = [[0, 1]]
        attractor_states = [[0, 1]]
        TP, FP, TN, FN = get_reachability(answer, guess, get_type="attractor", scc_indices=scc_indices, attractor_states=attractor_states)
        self.assertEqual(TP, 2)
        self.assertEqual(FP, 0)
        self.assertEqual(TN, 2)
        self.assertEqual(FN, 0)

    def test_type_parameter_with_basin_and_correct_scc_indices_and_attractor_states(self):
        answer = np.array([[0, 0, 1], [0, 1, 0], [0, 0, 1]])
        guess = np.array([[0, 0, 1], [0, 1, 0], [0, 0, 1]])
        scc_indices = [[0], [1, 2]]
        attractor_states = [[1, 2]]
        TP, FP, TN, FN = get_reachability(answer, guess, get_type="basin", scc_indices=scc_indices, attractor_states=attractor_states)
        self.assertEqual(TP, 1)
        self.assertEqual(FP, 0)
        self.assertEqual(TN, 1)
        self.assertEqual(FN, 0)

    def test_example(self):
        answer = np.array([[  0,   0,   0,   1],
                           [  0, 1/2, 1/2,   0],
                           [  0, 1/2, 1/2,   0],
                           [  0,   0,   0,   1]])
        # larger attractor
        guess1 = np.array([[1/3, 1/3, 1/3,   0],
                           [1/3, 1/3, 1/3,   0],
                           [1/3, 1/3, 1/3,   0],
                           [  0,   0,   0,   1]])
        # missing attractor
        guess2 = np.array([[  0,   0,   0,   1],
                           [  0,   0,   0,   1],
                           [  0,   0,   0,   1],
                           [  0,   0,   0,   1]])
        # wrong basin
        guess3 = np.array([[  0, 1/4, 1/4, 1/2],
                           [  0, 1/2, 1/2,   0],
                           [  0, 1/2, 1/2,   0],
                           [  0,   0,   0,   1]])
        scc_indices = [[0], [1, 2], [3]]
        attractor_states = [[1, 2], [3]]
        TP, FP, TN, FN = get_reachability(answer, guess1, get_type="all", scc_indices=scc_indices, attractor_states=attractor_states)
        self.assertEqual((TP, FP, TN, FN), (5, 5, 5, 1))
        TP, FP, TN, FN = get_reachability(answer, guess1, get_type="attractor", scc_indices=scc_indices, attractor_states=attractor_states)
        self.assertEqual((TP, FP, TN, FN), (5, 2, 5, 0))
        TP, FP, TN, FN = get_reachability(answer, guess1, get_type="basin", scc_indices=scc_indices, attractor_states=attractor_states)
        self.assertEqual((TP, FP, TN, FN), (0, 2, 0, 1))
        TP, FP, TN, FN = get_reachability(answer, guess2, get_type="all", scc_indices=scc_indices, attractor_states=attractor_states)
        self.assertEqual((TP, FP, TN, FN), (2, 2, 8, 4))
        TP, FP, TN, FN = get_reachability(answer, guess2, get_type="attractor", scc_indices=scc_indices, attractor_states=attractor_states)
        self.assertEqual((TP, FP, TN, FN), (1, 2, 5, 4))
        TP, FP, TN, FN = get_reachability(answer, guess2, get_type="basin", scc_indices=scc_indices, attractor_states=attractor_states)
        self.assertEqual((TP, FP, TN, FN), (1, 0, 2, 0))
        TP, FP, TN, FN = get_reachability(answer, guess3, get_type="all", scc_indices=scc_indices, attractor_states=attractor_states)
        self.assertEqual((TP, FP, TN, FN), (6, 2, 8, 0))
        TP, FP, TN, FN = get_reachability(answer, guess3, get_type="attractor", scc_indices=scc_indices, attractor_states=attractor_states)
        self.assertEqual((TP, FP, TN, FN), (5, 0, 7, 0))
        TP, FP, TN, FN = get_reachability(answer, guess3, get_type="basin", scc_indices=scc_indices, attractor_states=attractor_states)
        self.assertEqual((TP, FP, TN, FN), (1, 2, 0, 0))

    def test_hierarchy(self):
        answer = np.array([[1/2,   0,   0, 1/2],
                           [  0, 1/2, 1/2,   0],
                           [  0, 1/2, 1/2,   0],
                           [  0,   0,   0,   1]])
        # larger attractor
        guess1 = np.array([[1/3, 1/3, 1/3,   0],
                           [1/3, 1/3, 1/3,   0],
                           [1/3, 1/3, 1/3,   0],
                           [  0,   0,   0,   1]])
        # missing attractor
        guess2 = np.array([[  0,   0,   0,   1],
                           [  0,   0,   0,   1],
                           [  0,   0,   0,   1],
                           [  0,   0,   0,   1]])
        scc_indices = [[0], [1, 2], [3]]
        TP, FP, TN, FN = get_reachability(answer, guess1, get_type="hierarchy", scc_indices=scc_indices)
        self.assertEqual((TP, FP, TN, FN), (0, 2, 3, 0))
        TP, FP, TN, FN = get_reachability(answer, guess2, get_type="hierarchy", scc_indices=scc_indices)
        self.assertEqual((TP, FP, TN, FN), (0, 0, 5, 0))
    
    def test_hierarchy_debug(self):
        answer = np.array([[1/2,   0,   0, 1/2],
                           [  0, 1/2, 1/2,   0],
                           [  0, 1/2, 1/2,   0],
                           [1/2,   0,   0, 1/2]])
        # larger attractor
        guess = np.array([[1/3, 1/3, 1/3,   0],
                          [1/3, 1/3, 1/3,   0],
                          [1/3, 1/3, 1/3,   0],
                          [  0,   0,   0,   1]])
        scc_indices = [[0], [1, 2], [3]]
        with self.assertRaises(ValueError):
            get_reachability(answer, guess, get_type="hierarchy", scc_indices=scc_indices, DEBUG=True)

class TestIsBlockTriangular(unittest.TestCase):
    def test_block_triangular(self):
        transition_matrix = np.array([[0.5, 0.5, 0], [0.5, 0.5, 0], [0, 0, 1]])
        scc_indices = [[0, 1], [2]]
        self.assertTrue(is_block_triangular(transition_matrix, scc_indices))

    def test_non_matching_scc(self):
        transition_matrix = np.array([[0.5, 0.5, 0], [0.5, 0.5, 0], [0, 0, 1]])
        scc_indices = [[0], [1, 2]]
        self.assertFalse(is_block_triangular(transition_matrix, scc_indices))

    def test_single_scc(self):
        transition_matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
        scc_indices = [[0, 1]]
        self.assertTrue(is_block_triangular(transition_matrix, scc_indices))

    def test_multiple_scc(self):
        transition_matrix = np.array([[0.5, 0.5, 0, 0], [0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5], [0, 0, 0.5, 0.5]])
        scc_indices = [[0, 1], [2, 3]]
        self.assertTrue(is_block_triangular(transition_matrix, scc_indices))

    def test_empty_matrix(self):
        transition_matrix = np.array([])
        scc_indices = []
        self.assertTrue(is_block_triangular(transition_matrix, scc_indices))


class TestGetBlockTriangular(unittest.TestCase):
    def test_transition_matrix_only(self):
        transition_matrix = np.array([[0.5,   0, 0, 0.5],
                                      [  0, 0.5, 0, 0.5],
                                      [  0, 0.5, 0, 0.5],
                                      [0.5,   0, 0, 0.5]])
        expected_block_triangular = np.array([[0, 0.5,   0, 0.5],
                                              [0, 0.5,   0, 0.5],
                                              [0,   0, 0.5, 0.5],
                                              [0,   0, 0.5, 0.5]])
        expected_scc_indices = [[2], [1], [0, 3]]
        block_triangular, scc_indices = get_block_triangular(transition_matrix)
        self.assertTrue(np.allclose(block_triangular, expected_block_triangular))
        self.assertEqual(scc_indices, expected_scc_indices)

    def test_transition_matrix_and_scc_indices(self):
        transition_matrix = np.array([[0.5,   0, 0, 0.5],
                                      [  0, 0.5, 0, 0.5],
                                      [  0, 0.5, 0, 0.5],
                                      [0.5,   0, 0, 0.5]])
        scc_indices = [[2], [1], [0, 3]]
        expected_block_triangular = np.array([[0, 0.5,   0, 0.5],
                                              [0, 0.5,   0, 0.5],
                                              [0,   0, 0.5, 0.5],
                                              [0,   0, 0.5, 0.5]])
        block_triangular, _ = get_block_triangular(transition_matrix, scc_indices=scc_indices)
        self.assertTrue(np.allclose(block_triangular, expected_block_triangular))

    def test_transition_matrix_and_scc_dag(self):
        transition_matrix = np.array([[0.5,   0, 0, 0.5],
                                      [  0, 0.5, 0, 0.5],
                                      [  0, 0.5, 0, 0.5],
                                      [0.5,   0, 0, 0.5]])
        expected_block_triangular = np.array([[0, 0.5,   0, 0.5],
                                              [0, 0.5,   0, 0.5],
                                              [0,   0, 0.5, 0.5],
                                              [0,   0, 0.5, 0.5]])        
        stg = get_stg(transition_matrix)
        scc_dag = get_scc_dag(stg)
        block_triangular, _ = get_block_triangular(transition_matrix, scc_dag=scc_dag)
        self.assertTrue(np.allclose(block_triangular, expected_block_triangular))

    def test_transition_matrix_and_stg(self):
        transition_matrix = np.array([[0.5,   0, 0, 0.5],
                                      [  0, 0.5, 0, 0.5],
                                      [  0, 0.5, 0, 0.5],
                                      [0.5,   0, 0, 0.5]])
        expected_block_triangular = np.array([[0, 0.5,   0, 0.5],
                                              [0, 0.5,   0, 0.5],
                                              [0,   0, 0.5, 0.5],
                                              [0,   0, 0.5, 0.5]])
        stg = get_stg(transition_matrix)
        block_triangular, _ = get_block_triangular(transition_matrix, stg=stg)
        self.assertTrue(np.allclose(block_triangular, expected_block_triangular))

    def test_all_inputs(self):
        transition_matrix = np.array([[0.5,   0, 0, 0.5],
                                      [  0, 0.5, 0, 0.5],
                                      [  0, 0.5, 0, 0.5],
                                      [0.5,   0, 0, 0.5]])
        expected_block_triangular = np.array([[0, 0.5,   0, 0.5],
                                              [0, 0.5,   0, 0.5],
                                              [0,   0, 0.5, 0.5],
                                              [0,   0, 0.5, 0.5]])
        scc_indices = [[2], [1], [0, 3]]
        stg = get_stg(transition_matrix)
        scc_dag = get_scc_dag(stg)
        block_triangular, _ = get_block_triangular(transition_matrix, scc_indices=scc_indices, scc_dag=scc_dag, stg=stg)
        self.assertTrue(np.allclose(block_triangular, expected_block_triangular))

    def test_invalid_inputs(self):
        transition_matrix = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])  # non-square matrix
        with self.assertRaises(ValueError):
            get_block_triangular(transition_matrix, DEBUG=True)

    def test_debug(self):
        transition_matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
        scc_indices = [[0], [1]]
        get_block_triangular(transition_matrix, scc_indices=scc_indices, DEBUG=True)


class TestEnforceAsynchronous(unittest.TestCase):
    def test_matrix_shape(self):
        transition_matrix = np.random.rand(4, 4)
        result = enforce_asynchronous(transition_matrix)
        self.assertTrue(np.allclose(result.shape, transition_matrix.shape))

    def test_2x2_matrix(self):
        transition_matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
        result = enforce_asynchronous(transition_matrix)
        self.assertTrue(np.allclose(result, transition_matrix))

    def test_4x4_matrix(self):
        transition_matrix = np.array([[1/4, 1/4, 1/4, 1/4],
                                      [1/4, 1/4, 1/4, 1/4],
                                      [1/4, 1/4, 1/4, 1/4],
                                      [1/4, 1/4, 1/4, 1/4]])
        result = enforce_asynchronous(transition_matrix)
        expected_result = np.array([[1/3, 1/3, 1/3,   0],
                                    [1/3, 1/3,   0, 1/3],
                                    [1/3,   0, 1/3, 1/3],
                                    [  0, 1/3, 1/3, 1/3]])
        self.assertTrue(np.allclose(result, expected_result))

    def test_non_square_matrix(self):
        transition_matrix = np.random.rand(3, 4)
        with self.assertRaises(ValueError):
            enforce_asynchronous(transition_matrix, DEBUG=True)

    def test_non_binary_values(self):
        transition_matrix = np.array([[0.5, 0.5], [0.5, 1.5]])
        with self.assertRaises(ValueError):
            enforce_asynchronous(transition_matrix, DEBUG=True)

if __name__ == '__main__':
    unittest.main()