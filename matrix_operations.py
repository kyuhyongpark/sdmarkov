import numpy as np

from graph import get_stg
from transition_matrix import check_transition_matrix, get_hamming_distance_matrix
from scc_dags import get_scc_dag, get_scc_states


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
        If the matrix is not square, or if the elements of the matrix are not
        between 0 and 1, or if the rows of the matrix do not sum to 1.
    """

    if DEBUG:
        check_transition_matrix(matrix, compressed=True)

    squared_matrix = matrix.astype(np.float64)

    for _ in range(n):
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
        assert np.allclose(squared_matrix, double_squared_matrix), (
            f"The result using n+1 {double_squared_matrix} is not close to the "
            f"result using n {squared_matrix}."
        )

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
        check_transition_matrix(matrix)

        # Check that index groups are mutually exclusive
        for i in range(len(index_groups)):
            for j in range(i + 1, len(index_groups)):
                if set(index_groups[i]).intersection(set(index_groups[j])):
                    raise ValueError("Index groups must be mutually exclusive.")

        # Check that index groups are within the bounds of the matrix
        for group in index_groups:
            if any(index < 0 or index >= matrix.shape[0] for index in group):
                raise ValueError("Index groups must be within the bounds of the matrix.")

    all_indices = []
    for group in index_groups:
        all_indices.extend(group)

    row_merged = np.delete(matrix, all_indices, axis=0)

    for group in reversed(index_groups):
        if len(group) == 0:
            continue
        insert_row = np.mean(matrix[group], axis=0)
        row_merged = np.insert(row_merged, 0, insert_row, axis=0)

    merged = np.delete(row_merged, all_indices, axis=1)

    for group in reversed(index_groups):
        if len(group) == 0:
            continue
        insert_column = np.sum(row_merged[:, group], axis=1)
        merged = np.insert(merged, 0, insert_column, axis=1)

    # Ensure that the matrix is stochastic
    merged[merged < 0] = 0

    merged /= np.sum(merged, axis=1, keepdims=True)

    if DEBUG:
        # get the number of non-empty index groups
        non_empty_groups = [group for group in index_groups if group]

        # Check that the result is a matrix of dimension length(non_empty_groups) x length(non_empty_groups)
        assert merged.shape == (
            len(non_empty_groups),
            len(non_empty_groups),
        ), f"The result {merged} is not a matrix of dimension {len(non_empty_groups)} x {len(non_empty_groups)}."

        check_transition_matrix(merged, compressed=True)

    return merged


def expand_matrix(
    matrix: np.ndarray, index_groups: list[list[int]], DEBUG: bool = False
) -> np.ndarray:
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
        check_transition_matrix(matrix, compressed=True)

        # Check that index groups are mutually exclusive
        for i in range(len(index_groups)):
            for j in range(i + 1, len(index_groups)):
                if set(index_groups[i]).intersection(set(index_groups[j])):
                    raise ValueError("Index groups must be mutually exclusive.")

        # Check that the number of non-empty index groups is not greater than the size of the matrix
        non_empty_groups = [group for group in index_groups if group]
        if len(non_empty_groups) > matrix.shape[0]:
            raise ValueError(
                "The number of non-empty index groups must be less than or equal to the size of the matrix."
            )

    compressed_matrix_dimension = len(matrix)

    all_indices = []
    for group in index_groups:
        all_indices.extend(group)

    non_empty_groups = [group for group in index_groups if group]

    expanded_matrix_dimension = (
        compressed_matrix_dimension - len(non_empty_groups) + len(all_indices)
    )

    row_expanded = np.zeros((expanded_matrix_dimension, compressed_matrix_dimension))

    j = 0
    for i in range(expanded_matrix_dimension):
        if i not in all_indices:
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
        if i not in all_indices:
            # First len(non_empty_groups) columns of the matrix are the compressed columns,
            # and the rest are the not-compressed columns.
            # Here restore the j-th not-compressed column.
            expanded[:, i] = row_expanded[:, len(non_empty_groups) + j]
            j += 1
        else:
            for k, group in enumerate(non_empty_groups):
                if i in group:
                    expanded[:, i] = row_expanded[:, k] / len(group)
                    break

    # Ensure that the matrix is stochastic
    expanded[expanded < 0] = 0
    expanded /= np.sum(expanded, axis=1, keepdims=True)

    return expanded


def get_rms_diff(A: np.ndarray, B: np.ndarray, compressed: bool = False, partial: bool = False, row_wise_average: bool = False, DEBUG: bool = False) -> float:
    """
    Calculate the root mean squared (RMS) difference between two stochastic matrices A and B.

    Parameters
    ----------
    A : numpy array
        The first matrix.
    B : numpy array
        The second matrix.
    compressed : bool, optional
        If True, the matrices are compressed,
        and the number of rows/columns don't need to be 2^N.
    partial : bool, optional
        If True, the matrices are partial,
        and number of rows/columns don't need to be 2^N,
        the matrixes don't need to be square,
        and each row doesn't need to sum to 1.
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

    Note
    ----
    As the size of each row of the stochastic matrix grows, the RMS difference will approach 0.
    """

    if DEBUG:
        # Check that the matrices have the same shape
        if A.shape != B.shape:
            raise ValueError("The matrices must have the same shape.")

        check_transition_matrix(A, compressed, partial)
        check_transition_matrix(B, compressed, partial)

    # Convert A and B to floats
    A = A.astype(np.float64)
    B = B.astype(np.float64)

    if not row_wise_average:
        # Calculate the squared difference
        diff_squared = (A - B)**2

        # Calculate the mean of the squared difference
        mse = np.mean(diff_squared)

        # Calculate the RMS difference
        rms_diff = np.sqrt(mse)

        return rms_diff

    else:
        # Calculate the squared difference
        diff_squared = (A - B)**2

        # Calculate the mean of the squared difference
        mse = np.mean(diff_squared, axis=1)

        # Calculate the RMS difference
        rms_diff = np.sqrt(mse)

        return np.mean(rms_diff)


def get_dkl(A: np.ndarray, B: np.ndarray, compressed: bool = False, partial: bool = False, row_wise_average: bool = False, DEBUG: bool = False) -> float:
    """
    Calculate the Kullback-Leibler divergence between two stochastic matrices A and B.

    Parameters
    ----------
    A : numpy array
        The first matrix.
    B : numpy array
        The second matrix.
    compressed : bool, optional
        If True, the matrices are compressed,
        and the number of rows/columns don't need to be 2^N.
    partial : bool, optional
        If True, the matrices are partial,
        and number of rows/columns don't need to be 2^N,
        the matrixes don't need to be square,
        and each row doesn't need to sum to 1.
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

        # The dimension of the matricies should be 2
        if A.ndim != 2:
            raise ValueError("The matrices must be 2-dimensional.")

        check_transition_matrix(A, compressed, partial)
        check_transition_matrix(B, compressed, partial)

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

    if not row_wise_average:
        # Calculate the total KL divergence
        total_dkl = np.sum(dkl)

        return total_dkl

    else:
        # Calculate the average KL divergence
        average_dkl = np.mean(dkl)

        return average_dkl


def get_confusion_matrix(
    answer: np.ndarray,
    guess: np.ndarray,
    compressed: bool = False,
    partial: bool = False,
    DEBUG: bool = False
) -> tuple[int, int, int, int]:
    """
    Calculate the true positives, false positives, true negatives, and false negatives
    between two matrices, `answer` and `guess`.

    Parameters
    ----------
    answer : np.ndarray
        The ground truth matrix.
    guess : np.ndarray
        The predicted matrix.
    compressed : bool, optional
        If True, the matrices are compressed,
        and the number of rows/columns don't need to be 2^N.
    partial : bool, optional
        If True, the matrices are partial,
        and number of rows/columns don't need to be 2^N,
        the matrixes don't need to be square,
        and each row doesn't need to sum to 1.
    DEBUG : bool, optional
        If True, perform basic checks.

    Returns
    -------
    tuple
        A tuple containing four integers: (TP, FP, TN, FN)
        - TP: Number of true positives
        - FP: Number of false positives
        - TN: Number of true negatives
        - FN: Number of false negatives

    Examples
    --------
    >>> answer = np.array([[1, 0], [0, 1]])
    >>> guess = np.array([[1, 0], [1/2, 1/2]])
    >>> get_confusion_matrix(answer, guess)
    (2, 1, 1, 0)
    """

    if DEBUG:
        # Check that the matrices have the same shape
        if answer.shape != guess.shape:
            raise ValueError("The matrices must have the same shape.")

        check_transition_matrix(answer, compressed, partial)
        check_transition_matrix(guess, compressed, partial)

    # Define the conditions for each category
    TP = np.sum((answer != 0) & (guess != 0))  # True positives
    FP = np.sum((answer == 0) & (guess != 0))  # False positives
    TN = np.sum((answer == 0) & (guess == 0))  # True negatives
    FN = np.sum((answer != 0) & (guess == 0))  # False negatives

    if DEBUG:
        # Check that the sum of TP, FP, TN, and FN is equal to the total number of elements
        if TP + FP + TN + FN != answer.size:
            raise ValueError("The sum of TP, FP, TN, and FN must be equal to the total number of elements.")

    return TP, FP, TN, FN


def get_reachability(answer, guess, get_type="all", scc_indices=None, attractor_states=None, DEBUG=False):
    """
    Calculate the true positives, false positives, true negatives, and false negatives
    between two matrices, `answer` and `guess`.

    Parameters
    ----------
    answer : np.ndarray
        The ground truth matrix.
    guess : np.ndarray
        The predicted matrix.
    get_type : str, optional
        What type of reachability to calculate. Can be 'all', 'attractor', or 'basin'.
    scc_indices : list[list[int]], optional
        The indices of the strongly connected components in the transition matrix.
    attractor_states : list[list[int]], optional
        The indices of the attractor states in the transition matrix.
    DEBUG : bool, optional
        If True, perform basic checks.

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
    'all' type considers all reachability.
    'hierarchy' type considers only reachability from transient states, which should not exist in the answer matrix.
    'attractor' type considers only reachability from attractor states.
    'basin' type considers only reachability from transient states to attractor states.
    """

    if get_type not in ["all", "hierarchy", "attractor", "basin"]:
        raise ValueError("type must be one of 'all', 'hierarchy', 'attractor', or 'basin'.")

    if DEBUG:
        # Check that the matrices have the same shape
        if answer.shape != guess.shape:
            raise ValueError("The matrices must have the same shape.")

        check_transition_matrix(answer)
        check_transition_matrix(guess)

    if get_type == "hierarchy":
        if scc_indices is None:
            raise ValueError("If type is 'hierarchy', scc_indices must be provided.")

        block_answer, _ = get_block_triangular(answer, scc_indices=scc_indices, DEBUG=DEBUG)
        block_guess, _ = get_block_triangular(guess, scc_indices=scc_indices, DEBUG=DEBUG)

        length_list = [len(scc) for scc in scc_indices]

        answer = []
        guess = []

        current_index = 0
        for length in length_list:
            # Check that the first length columns are zeros from the length+1th row
            for i in range(current_index + length, block_answer.shape[0]):
                for j in range(current_index, current_index + length):
                    answer.append(block_answer[i, j])
                    guess.append(block_guess[i, j])

            current_index += length

        answer = np.array(answer)
        guess = np.array(guess)

        if DEBUG:
            # Check that answer is all zeros
            if np.any(answer):
                raise ValueError("The answer matrix must be all zeros.")

    elif get_type == "attractor":
        if scc_indices is None or attractor_states is None:
            raise ValueError("If type is 'attractor', scc_indices and attractor_states must be provided.")

        block_answer, _ = get_block_triangular(answer, scc_indices=scc_indices, DEBUG=DEBUG)
        block_guess, _ = get_block_triangular(guess, scc_indices=scc_indices, DEBUG=DEBUG)

        all_attractor_states = []
        for attractor_state in attractor_states:
            all_attractor_states.extend(attractor_state)

        answer = block_answer[-len(all_attractor_states):, :]
        guess = block_guess[-len(all_attractor_states):, :]

    elif get_type == "basin":
        if scc_indices is None or attractor_states is None:
            raise ValueError("If type is 'basin', scc_indices and attractor_states must be provided.")

        block_answer, _ = get_block_triangular(answer, scc_indices=scc_indices, DEBUG=DEBUG)
        block_guess, _ = get_block_triangular(guess, scc_indices=scc_indices, DEBUG=DEBUG)

        all_attractor_states = []
        for attractor_state in attractor_states:
            all_attractor_states.extend(attractor_state)

        answer = block_answer[:-len(all_attractor_states), -len(all_attractor_states):]
        guess = block_guess[:-len(all_attractor_states), -len(all_attractor_states):]

    # Define the conditions for each category
    TP = np.sum((answer != 0) & (guess != 0))  # True positives
    FP = np.sum((answer == 0) & (guess != 0))  # False positives
    TN = np.sum((answer == 0) & (guess == 0))  # True negatives
    FN = np.sum((answer != 0) & (guess == 0))  # False negatives

    return TP, FP, TN, FN


def is_block_triangular(transition_matrix: np.ndarray, scc_indices: list[list[int]]) -> bool:
    """
    Check if the transition matrix is block triangular according to the given scc indices.

    Parameters
    ----------
    transition_matrix : np.ndarray, shape (2^N, 2^N)
        The transition matrix.
    scc_indices : list[list[int]]
        The indices of the strongly connected components in the transition matrix.

    Returns
    -------
    bool
        True if the transition matrix is block triangular according to the scc indices, False otherwise.
    """

    length_list = [len(scc) for scc in scc_indices]

    current_index = 0
    for length in length_list:
        # Check that the first length columns are zeros from the length+1th row
        for i in range(current_index + length, transition_matrix.shape[0]):
            for j in range(current_index, current_index + length):
                if transition_matrix[i, j] != 0:
                    print(
                        f"Value at ({i}, {j}) is not zero and should be zero: {transition_matrix[i, j]}"
                    )
                    return False

        current_index += length

    return True


def get_block_triangular(transition_matrix, scc_indices=None, scc_dag=None, stg=None, DEBUG=False):
    """
    Get the block triangular matrix corresponding to the SCC DAG.

    Parameters
    ----------
    transition_matrix : numpy array, shape (2^N, 2^N)
        The transition matrix.
    scc_indices : list[list[int]], optional
        The indices of the strongly connected components in the stg.
        If None, it will be computed from the appropriate input.
    scc_dag : networkx DiGraph, optional
        The SCC DAG.
    stg : networkx DiGraph, optional
        The state transition graph.
    DEBUG : bool, optional
        If True, performs additional checks.

    Returns
    -------
    block_triangular : numpy array, shape (2^N, 2^N)
        The block triangular matrix corresponding to the SCC DAG.
    scc_indices : list[list[int]]
        The indices of the strongly connected components in the stg.

    Notes
    -----
    Note that the topological order of the scc dag is not unique.
    Use scc_indices when comparing different block triangular matrices,
    so that the order of states is consistent.

    Also note that the outcome may not be block triangular
    if using scc_indices of a different matrix.
    """

    if DEBUG:
        check_transition_matrix(transition_matrix)

    # Starting from the transition matrix
    if stg == None and scc_dag == None and scc_indices == None:
        stg = get_stg(transition_matrix)

    # Starting from the stg
    if stg != None and scc_dag == None and scc_indices == None:
        scc_dag = get_scc_dag(stg)

    # Starting from the scc dag
    if scc_dag != None and scc_indices == None:
        scc_indices = get_scc_states(scc_dag, as_indices=True)
        if DEBUG:
            print("Calculated scc_indices", scc_indices)

    # Starting from the scc indices
    index_list = []
    for scc in scc_indices:
        index_list.extend(scc)
    
    block_triangular = reorder_matrix(transition_matrix, index_list)

    return block_triangular, scc_indices


def enforce_asynchronous(transition_matrix, DEBUG=False):
    """
    Enforce asynchronous updates in a transition matrix.

    Parameters
    ----------
    transition_matrix : numpy array, shape (2^N, 2^N)
        The transition matrix.

    Returns
    -------
    transition_matrix : numpy array, shape (2^N, 2^N)
        The transition matrix with asynchronous updates enforced.

    Notes
    -----
    This function enforces asynchronous updates in a transition matrix by setting all
    transition probabilities to 0 if the Hamming distance between the two states is
    higher than 1. The transition matrix is then normalized.
    """

    if DEBUG:
        check_transition_matrix(transition_matrix)

    N = int(np.log2(transition_matrix.shape[0]))

    hd = get_hamming_distance_matrix(N, DEBUG=True)

    # for every i,j in the transition matrix, if the hamming distance is higher than 1, set it to 0
    for i in range(transition_matrix.shape[0]):
        for j in range(transition_matrix.shape[1]):
            if hd[i, j] > 1:
                transition_matrix[i, j] = 0

    # normalize the transition matrix
    transition_matrix = transition_matrix / np.sum(transition_matrix, axis=1, keepdims=True)

    return transition_matrix
