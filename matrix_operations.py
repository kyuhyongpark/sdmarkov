import numpy as np

def reorder_matrix(matrix, index_list):
    # Check if the length of index_list matches the dimensions of the matrix
    if len(index_list) != matrix.shape[0] or len(index_list) != matrix.shape[1]:
        raise ValueError("The length of index_list must match the size of the matrix (n x n).")
    
    # Reorder rows and columns according to the index list
    reordered_matrix = matrix[index_list, :][:, index_list]
    return reordered_matrix


def npower(matrix, n):
    return np.linalg.matrix_power(matrix, n)


# given a numpy array, merge certain rows and columns into one row and one column
def compress_matrix(A, index_groups):

    # print(A)

    all_indexes = []
    for group in index_groups:
        all_indexes.extend(group)

    row_merged = np.delete(A, all_indexes, axis=0)

    for group in reversed(index_groups):
        insert_row = np.mean(A[group], axis=0)
        row_merged = np.insert(row_merged, 0, insert_row, axis=0)

    merged = np.delete(row_merged, all_indexes, axis=1)

    for group in reversed(index_groups):
        insert_column = np.sum(row_merged[:,group], axis=1)
        merged = np.insert(merged, 0, insert_column, axis=1)

    # print(merged)

    return merged


def expand_matrix(A, index_groups):

    compressed_matrix_dimension = len(A)
    
    all_indexes = []
    for group in index_groups:
        all_indexes.extend(group)

    expanded_matrix_dimension = compressed_matrix_dimension - len(index_groups) + len(all_indexes)

    row_expanded = np.zeros((expanded_matrix_dimension, compressed_matrix_dimension))

    j = 0
    for i in range(expanded_matrix_dimension):
        if i not in all_indexes:
            row_expanded[i] = A[len(index_groups) + j]
            j += 1
        else:
            for k, group in enumerate(index_groups):
                if i in group:
                    row_expanded[i] = A[k]
                    break

    # print(row_expanded)

    expanded = np.zeros((expanded_matrix_dimension, expanded_matrix_dimension))

    j = 0
    for i in range(expanded_matrix_dimension):
        if i not in all_indexes:
            expanded[:,i] = row_expanded[:,len(index_groups) + j]
            j += 1
        else:
            for k, group in enumerate(index_groups):
                if i in group:
                    expanded[:,i] = row_expanded[:,k]/len(group)
                    break

    # print(np.round(expanded, 3))

    return expanded


def get_rms_diff(A, B):
    # Calculate the squared difference
    diff_squared = (A - B)**2

    # Calculate the mean of the squared difference
    mse = np.mean(diff_squared)

    # Calculate the RMS difference
    rms_diff = np.sqrt(mse)

    return rms_diff


def get_dkl(A, B):

    ratio = np.divide(A,B, out=np.zeros_like(A), where=A!=0)

    # print(ratio)

    lose_information = A * np.log(ratio, out=np.zeros_like(A), where=ratio!=0)
    
    # print(lose_information)

    total_lost = np.sum(lose_information)

    return total_lost