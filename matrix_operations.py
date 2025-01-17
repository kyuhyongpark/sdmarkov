import numpy as np

def reorder_matrix(matrix: np.ndarray, index_list: list[int]) -> np.ndarray:
    """
    Reorders the rows and columns of a matrix according to the given index list.

    Parameters
    ----------
    matrix : numpy.ndarray
        The matrix to be reordered.
    index_list : list[int]
        A list of the new order of the rows and columns of the matrix.

    Returns
    -------
    numpy.ndarray
        The reordered matrix.

    Raises
    ------
    ValueError
        If the length of index_list does not match the size of the matrix (n x n).

    Examples
    --------
    >>> m = np.array([[1,2,3], [4,5,6], [7,8,9]])
    >>> reorder_matrix(m, [2,1,0])
    array([[9, 8, 7],
           [6, 5, 4],
           [3, 2, 1]])
    """
    # Check if the length of index_list matches the dimensions of the matrix
    if len(index_list) != matrix.shape[0] or len(index_list) != matrix.shape[1]:
        raise ValueError("The length of index_list must match the size of the matrix (n x n).")
    
    # Reorder rows and columns according to the index list
    reordered_matrix = matrix[index_list, :][:, index_list]
    return reordered_matrix


def npower(matrix, n):
    return np.linalg.matrix_power(matrix, n)


def compress_matrix(
    matrix: np.ndarray,
    index_groups: list[list[int]],
    DEBUG: bool = False,
) -> np.ndarray:
    """
    Compress a matrix by merging specified rows and columns.

    Parameters
    ----------
    matrix : numpy.ndarray
        The original matrix to be compressed.
    index_groups : list[list[int]]
        A list of index groups, where each group specifies the rows and columns
        to be merged into a single row and column.
    DEBUG : bool, optional
        If True, perform basic checks.

    Returns
    -------
    numpy.ndarray
        The compressed matrix with merged rows and columns.

    Notes
    -----
    - Rows specified in the same index group are averaged into a single row.
    - Columns specified in the same index group are summed into a single column.

    Examples
    --------
    >>> matrix = np.array([[1, 0,   0  ],
    ...                    [0, 1/2, 1/2],
    ...                    [0, 0,   1  ]])
    >>> index_groups = [[0, 1], [2]]
    >>> compressed_matrix = compress_matrix(matrix, index_groups)
    >>> compressed_matrix
    array([[3/4,  1/4,],
           [0,    1,  ]])
    """

    if DEBUG:
        # Check that the matrix is square
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("The matrix must be square.")

        # Check that the elements of the array are between 0 and 1
        if not np.all(matrix >= 0) or not np.all(matrix <= 1):
            raise ValueError("All elements of the matrix must be between 0 and 1.")
        
        # Check that every row of the matrix sums to 1
        if not np.allclose(np.sum(matrix, axis=1), np.ones(matrix.shape[1])):
            raise ValueError("Every row of the matrix must sum to 1.")
        
        # Check that index groups are mutually exclusive
        for i in range(len(index_groups)):
            for j in range(i+1, len(index_groups)):
                if set(index_groups[i]).intersection(set(index_groups[j])):
                    raise ValueError("Index groups must be mutually exclusive.")
        
        # Check that index groups are within the bounds of the matrix
        for group in index_groups:
            if any(index < 0 or index >= matrix.shape[0] for index in group):
                raise ValueError("Index groups must be within the bounds of the matrix.")

    all_indexes = []
    for group in index_groups:
        all_indexes.extend(group)

    row_merged = np.delete(matrix, all_indexes, axis=0)

    for group in reversed(index_groups):
        insert_row = np.mean(matrix[group], axis=0)
        row_merged = np.insert(row_merged, 0, insert_row, axis=0)

    merged = np.delete(row_merged, all_indexes, axis=1)

    for group in reversed(index_groups):
        insert_column = np.sum(row_merged[:,group], axis=1)
        merged = np.insert(merged, 0, insert_column, axis=1)

    return merged


def expand_matrix(matrix: np.ndarray, index_groups: list[list[int]], DEBUG: bool = False) -> np.ndarray:
    """
    Expand a compressed matrix by splitting certain rows and columns into multiple rows and columns.

    Parameters
    ----------
    matrix : np.ndarray
        The compressed matrix to expand.
    index_groups : list[list[int]]
        A list of lists, same as the one used in the compress_matrix function.
    DEBUG : bool, optional
        If True, perform basic checks.
        
    Returns
    -------
    np.ndarray
        The expanded matrix.

    Examples
    --------
    >>> matrix = np.array([[3/4, 1/4], [0, 1]])
    >>> index_groups = [[0, 1], [2]]
    >>> expanded_matrix = expand_matrix(matrix, index_groups)
    >>> expanded_matrix
    array([[3/8, 3/8, 1/4],
           [3/8, 3/8, 1/4],
           [0,   0,   1  ]])
    """

    if DEBUG:
        # Check that the matrix is square
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("The matrix must be square.")

        # Check that the elements of the array are between 0 and 1
        if not np.all(matrix >= 0) or not np.all(matrix <= 1):
            raise ValueError("All elements of the matrix must be between 0 and 1.")
        
        # Check that every row of the matrix sums to 1
        if not np.allclose(np.sum(matrix, axis=1), np.ones(matrix.shape[1])):
            raise ValueError("Every row of the matrix must sum to 1.")
        
        # Check that index groups are mutually exclusive
        for i in range(len(index_groups)):
            for j in range(i+1, len(index_groups)):
                if set(index_groups[i]).intersection(set(index_groups[j])):
                    raise ValueError("Index groups must be mutually exclusive.")
        
        # Check that the number of index groups is not greater than the size of the matrix
        if len(index_groups) > matrix.shape[0]:
            raise ValueError("The number of index groups must be less than or equal to the size of the matrix.")
        

    compressed_matrix_dimension = len(matrix)
    
    all_indexes = []
    for group in index_groups:
        all_indexes.extend(group)

    expanded_matrix_dimension = compressed_matrix_dimension - len(index_groups) + len(all_indexes)

    row_expanded = np.zeros((expanded_matrix_dimension, compressed_matrix_dimension))

    j = 0
    for i in range(expanded_matrix_dimension):
        if i not in all_indexes:
            row_expanded[i] = matrix[len(index_groups) + j]
            j += 1
        else:
            for k, group in enumerate(index_groups):
                if i in group:
                    row_expanded[i] = matrix[k]
                    break

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

    return expanded


def get_rms_diff(A: np.ndarray, B: np.ndarray, DEBUG: bool = False) -> float:
    """
    Calculate the root mean squared (RMS) difference between two stochastic matrices A and B.

    Parameters
    ----------
    A : numpy array
        The first matrix.
    B : numpy array
        The second matrix.
    DEBUG : bool, optional
        If True, perform basic checks.

    Returns
    -------
    float
        The RMS difference between A and B.

    Examples
    --------
    >>> A = np.array([[1, 0], [0, 1]])
    >>> B = np.array([[1, 0], [1/2, 1/2]])
    >>> get_rms_diff(A, B)
    0.35355339059327376220042218105242
    """

    if DEBUG:
        # Check that the matrices have the same shape
        if A.shape != B.shape:
            raise ValueError("The matrices must have the same shape.")

        # Check that the elements of the array are between 0 and 1
        if not np.all(A >= 0) or not np.all(A <= 1) or not np.all(B >= 0) or not np.all(B <= 1):
            raise ValueError("All elements of the matrix must be between 0 and 1.")
        
        # Check that every row of the matrix sums to 1
        if not np.allclose(np.sum(A, axis=1), np.ones(A.shape[1])) or not np.allclose(np.sum(B, axis=1), np.ones(B.shape[1])):
            raise ValueError("Every row of the matrix must sum to 1.")

    # Convert A and B to floats
    A = A.astype(float)
    B = B.astype(float)

    # Calculate the squared difference
    diff_squared = (A - B)**2

    # Calculate the mean of the squared difference
    mse = np.mean(diff_squared)

    # Calculate the RMS difference
    rms_diff = np.sqrt(mse)

    return rms_diff


def get_dkl(A: np.ndarray, B: np.ndarray, DEBUG: bool = False) -> float:
    """
    Calculate the Kullback-Leibler divergence between two stochastic matrices A and B.

    Parameters
    ----------
    A : numpy array
        The first matrix.
    B : numpy array
        The second matrix.
    DEBUG : bool, optional
        If True, perform basic checks.

    Returns
    -------
    float
        The Kullback-Leibler divergence between A and B.

    Examples
    --------
    >>> A = np.array([[1, 0], [0, 1]])
    >>> B = np.array([[1, 0], [1/2, 1/2]])
    >>> get_dkl(A, B)
    0.69314718055994530941723212145818
    """

    if DEBUG:
        # Check that the matrices have the same shape
        if A.shape != B.shape:
            raise ValueError("The matrices must have the same shape.")

        # Check that the elements of the array are between 0 and 1
        if not np.all(A >= 0) or not np.all(A <= 1) or not np.all(B >= 0) or not np.all(B <= 1):
            raise ValueError("All elements of the matrix must be between 0 and 1.")
        
        # Check that every row of the matrix sums to 1
        if not np.allclose(np.sum(A, axis=1), np.ones(A.shape[1])) or not np.allclose(np.sum(B, axis=1), np.ones(B.shape[1])):
            raise ValueError("Every row of the matrix must sum to 1.")

    # Convert A and B to floats
    A = A.astype(float)
    B = B.astype(float)

    # Calculate the ratio
    ratio = np.divide(A,B, out=np.zeros_like(A, dtype=float), where=A!=0.0)

    # Calculate the lost information
    lost_information = A * np.log(ratio, out=np.zeros_like(A, dtype=float), where=ratio!=0.0)
    
    # Calculate the total lost information
    total_lost = np.sum(lost_information)

    return total_lost
