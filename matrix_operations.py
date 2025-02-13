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


def nsquare(matrix: np.ndarray, n: int, DEBUG: bool = False) -> np.ndarray:

    """
    TODO: For the cases where the matrix will not converge for large n,
    we need options to average the outcome over some value.

    Compute the nth power of a matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
        The matrix to be powered.
    n : int
        The matrix will be squared n times.
    DEBUG : bool
        If True, perform basic checks on the matrix.

    Returns
    -------
    powered_matrix : numpy.ndarray
        The result of raising the matrix to the nth power.

    Raises
    ------
    ValueError
        If the matrix is not square, or if the elements of the matrix are not between 0 and 1, or if the rows of the matrix do not sum to 1.
    """

    if DEBUG:
        # Check that the matrix is square
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("The matrix must be square.")

        # Check that the elements of the array are between 0 and 1, with some tolerance
        if not np.all(matrix >= 0 - 1e-16) or not np.all(matrix <= 1 + 1e-16):
            raise ValueError("All elements of the matrix must be between 0 and 1. Max: {}, Min: {}".format(np.max(matrix), np.min(matrix)))

        # Check that every row of the matrix sums to 1
        if not np.allclose(np.sum(matrix, axis=1), np.ones(matrix.shape[1])):
            raise ValueError("Every row of the matrix must sum to 1.")
        
        # Make the matrix datatype float64
        matrix = matrix.astype(np.float64)

    squared_matrix = matrix

    for i in range(n):
        squared_matrix = np.linalg.matrix_power(squared_matrix, 2)

        # Set small negative values to zero
        squared_matrix[squared_matrix < 0] = 0

        # Normalize the matrix so that the sum of each row is 1
        squared_matrix /= np.sum(squared_matrix, axis=1, keepdims=True)

    if DEBUG:
        # Check that the result using n+1 is close to the result using n
        double_squared_matrix = np.linalg.matrix_power(squared_matrix, 2)
        double_squared_matrix[double_squared_matrix < 0] = 0
        double_squared_matrix /= np.sum(double_squared_matrix, axis=1, keepdims=True)
        assert np.allclose(squared_matrix, double_squared_matrix), f"The result using n+1 {double_squared_matrix} is not close to the result using n {squared_matrix}."

    return squared_matrix

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
            raise ValueError("All elements of the matrix must be between 0 and 1. Max: {}, Min: {}".format(np.max(matrix), np.min(matrix)))
        
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
        if len(group) == 0:
            continue
        insert_row = np.mean(matrix[group], axis=0)
        row_merged = np.insert(row_merged, 0, insert_row, axis=0)

    merged = np.delete(row_merged, all_indexes, axis=1)

    for group in reversed(index_groups):
        if len(group) == 0:
            continue
        insert_column = np.sum(row_merged[:,group], axis=1)
        merged = np.insert(merged, 0, insert_column, axis=1)

    # Ensure that the matrix is stochastic
    merged[merged < 0] = 0

    merged /= np.sum(merged, axis=1, keepdims=True)

    if DEBUG:
        # get the number of non-empty index groups
        non_empty_groups = [group for group in index_groups if group]

        # Check that the result is a matrix of dimension length(non_empty_groups) x length(non_empty_groups)
        assert merged.shape == (len(non_empty_groups), len(non_empty_groups)), f"The result {merged} is not a matrix of dimension {len(non_empty_groups)} x {len(non_empty_groups)}."

        # Check that the elements of the array are between 0 and 1
        if not np.all(merged >= 0) or not np.all(merged <= 1):
            raise ValueError("All elements of the matrix must be between 0 and 1. Max: {}, Min: {}".format(np.max(merged), np.min(merged)))
        
        # Check that every row of the matrix sums to 1
        if not np.allclose(np.sum(merged, axis=1), np.ones(merged.shape[1])):
            raise ValueError("Every row of the matrix must sum to 1.")

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
            raise ValueError("All elements of the matrix must be between 0 and 1. Max: {}, Min: {}".format(np.max(matrix), np.min(matrix)))
        
        # Check that every row of the matrix sums to 1
        if not np.allclose(np.sum(matrix, axis=1), np.ones(matrix.shape[1])):
            raise ValueError("Every row of the matrix must sum to 1.")
        
        # Check that index groups are mutually exclusive
        for i in range(len(index_groups)):
            for j in range(i+1, len(index_groups)):
                if set(index_groups[i]).intersection(set(index_groups[j])):
                    raise ValueError("Index groups must be mutually exclusive.")
        
        # Check that the number of non-empty index groups is not greater than the size of the matrix
        non_empty_groups = [group for group in index_groups if group]
        if len(non_empty_groups) > matrix.shape[0]:
            raise ValueError("The number of non-empty index groups must be less than or equal to the size of the matrix.")
        

    compressed_matrix_dimension = len(matrix)
    
    all_indexes = []
    for group in index_groups:
        all_indexes.extend(group)

    non_empty_groups = [group for group in index_groups if group]

    expanded_matrix_dimension = compressed_matrix_dimension - len(non_empty_groups) + len(all_indexes)

    row_expanded = np.zeros((expanded_matrix_dimension, compressed_matrix_dimension))

    j = 0
    for i in range(expanded_matrix_dimension):
        if i not in all_indexes:
            # First len(non_empty_groups) rows of the matrix are the compressed rows,
            # and the rest are the not-compressed rows.
            # Here restore the j-th not-compressed row.
            row_expanded[i] = matrix[len(non_empty_groups) + j]
            j += 1
        else:
            for k, group in enumerate(non_empty_groups):
                if i in group:
                    row_expanded[i] = matrix[k]
                    break

    expanded = np.zeros((expanded_matrix_dimension, expanded_matrix_dimension))

    j = 0
    for i in range(expanded_matrix_dimension):
        if i not in all_indexes:
            # First len(non_empty_groups) columns of the matrix are the compressed columns,
            # and the rest are the not-compressed columns.
            # Here restore the j-th not-compressed column.
            expanded[:,i] = row_expanded[:,len(non_empty_groups) + j]
            j += 1
        else:
            for k, group in enumerate(non_empty_groups):
                if i in group:
                    expanded[:,i] = row_expanded[:,k]/len(group)
                    break


    # Ensure that the matrix is stochastic
    expanded[expanded < 0] = 0

    expanded /= np.sum(expanded, axis=1, keepdims=True)

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
    A = A.astype(np.float64)
    B = B.astype(np.float64)

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
    A = A.astype(np.float64)
    B = B.astype(np.float64)

    # Calculate the ratio
    ratio = np.divide(A,B, out=np.zeros_like(A, dtype=float), where=A!=0.0)

    # Calculate the lost information
    lost_information = A * np.log(ratio, out=np.zeros_like(A, dtype=float), where=ratio!=0.0)
    
    # Calculate the Kullback-Leibler divergence by summing over each row
    dkl = np.sum(lost_information, axis=1)

    # Each value in dkl should be non-negative
    dkl = np.maximum(dkl, 0.0)

    # Calculate the total KL divergence
    total_dkl = np.sum(dkl)

    return total_dkl


def get_reachability(answer, guess):
    """
    Calculate the true positives, false positives, true negatives, and false negatives
    between two matrices, `answer` and `guess`.

    Parameters
    ----------
    answer : np.ndarray
        The ground truth matrix.
    guess : np.ndarray
        The predicted matrix.

    Returns
    -------
    tuple
        A tuple containing four integers: (TP, FP, TN, FN)
        - TP: Number of true positives
        - FP: Number of false positives
        - TN: Number of true negatives
        - FN: Number of false negatives

    Notes
    -----
    - A true positive is counted when a corresponding element in both matrices is non-zero.
    - A false positive is counted when an element in `guess` is non-zero and the corresponding element in `answer` is zero.
    - A true negative is counted when both corresponding elements in `answer` and `guess` are zero.
    - A false negative is counted when an element in `answer` is non-zero and the corresponding element in `guess` is zero.
    """
    # TODO: Do so for different blocks

    # Define the conditions for each category
    TP = np.sum((answer != 0) & (guess != 0))  # True positives
    FP = np.sum((answer == 0) & (guess != 0))  # False positives
    TN = np.sum((answer == 0) & (guess == 0))  # True negatives
    FN = np.sum((answer != 0) & (guess == 0))  # False negatives

    return TP, FP, TN, FN