import numpy as np

from helper import check_transition_matrix


def get_convergence_matrix(
    T_inf_expanded: np.ndarray, 
    attractor_indices: list[list[int]], 
    DEBUG: bool = False
) -> np.ndarray:
    """
    Compute the convergence matrix that represents the probability of reaching each attractor from each state.

    Parameters
    ----------
    T_inf_expanded : np.ndarray
        The transition matrix at t=inf. Note that this should be expanded to 2**N x 2**N.
    attractor_indices : list of list of int
        The indices of the attractor states.
    DEBUG : bool, optional
        If True, performs additional checks.

    Returns
    -------
    convergence_matrix : np.ndarray
        A matrix representing the probability of reaching each attractor from each state.
    """

    if DEBUG:
        # Check that the given matrix is a transition matrix
        check_transition_matrix(T_inf_expanded)
    
        # Check that the attractor indices are valid
        for attractor in attractor_indices:
            for state in attractor:
                if 0 > state or state >= T_inf_expanded.shape[0]:
                    raise ValueError("The attractor indices must be valid.")

        # Check that the attractor indices are mutually exclusive
        for i in range(len(attractor_indices)):
            for j in range(i + 1, len(attractor_indices)):
                if set(attractor_indices[i]).intersection(set(attractor_indices[j])):
                    raise ValueError("The attractor indices must be mutually exclusive.")

    summed_columns = []
    for attractor in attractor_indices:
        summed = T_inf_expanded[:, attractor].sum(axis=1, keepdims=True)
        summed_columns.append(summed)

    convergence_matrix = np.hstack(summed_columns, dtype=np.float64)

    # Normalize the rows of the convergence matrix 
    for i in range(convergence_matrix.shape[0]):
        convergence_matrix[i, :] /= np.sum(convergence_matrix[i, :])


    if DEBUG:
        if convergence_matrix.shape[0] != len(T_inf_expanded):
            raise ValueError("The number of states does not match the number of rows in the convergence matrix.")
        if convergence_matrix.shape[1] != len(attractor_indices):
            raise ValueError("The number of attractors does not match the number of columns in the convergence matrix.")

    return convergence_matrix


def get_strong_basins(
    convergence_matrix: np.ndarray,
    DEBUG: bool = False,
) -> np.ndarray:
    """
    Computes whether each state belong to a strong basin.

    Parameters
    ----------
    convergence_matrix : numpy array
        A matrix that represents the probability of reaching each attractor from each state
    DEBUG : bool, optional
        If True, performs additional checks on the input data.

    Returns
    -------
    strong_basins : numpy array, shape (2**N, 1)
        If the state is in a strong basin, the value is 1.
        If the state is in a weak basin instead, the value is 0.
    """
    if DEBUG:
        check_transition_matrix(convergence_matrix, partial=True)

    strong_basin = np.zeros((convergence_matrix.shape[0], 1))
    for row in range(convergence_matrix.shape[0]):
        # if row has a single non-zero element
        if np.count_nonzero(convergence_matrix[row]) == 1:
            strong_basin[row] = 1

    return strong_basin


def get_basin_ratios(
    convergence_matrix: np.ndarray,
    DEBUG: bool = False,
) -> tuple[np.ndarray, list[list[str]]]:
    """
    Calculate the ratio of the size of each basin to the total number of states.

    Parameters
    ----------
    convergence_matrix : numpy array
        The probability of reaching each attractor from each state.
        Note that this should be 2**N x A.
    DEBUG : bool, optional
        If True, performs additional checks.

    Returns
    -------
    basin_ratio : 2D numpy array
        The probability of reaching each attractor.
        Note that it is 2D for compatibility with other functions.
        Each row must sum to 1.
    """

    if DEBUG:
        check_transition_matrix(convergence_matrix, partial=True)

    basin_ratios = np.mean(convergence_matrix, axis=0, keepdims=True)

    # Normalize the basin ratios
    basin_ratios /= basin_ratios.sum()

    return basin_ratios


def get_node_average_values(
    T_inf_expanded: np.ndarray,
    DEBUG: bool = False
) -> np.ndarray:
    """
    Compute the average value of each node, given the transition matrix at t=inf.

    Parameters
    ----------
    T_inf_expanded : numpy array
        The transition matrix at t=inf. Note that this should be 2**N x 2**N.
    DEBUG : bool, optional
        If True, performs additional checks.

    Returns
    -------
    node_average_values : numpy array, shape (1, N)
        The average value of each node.
    """

    if DEBUG:
        check_transition_matrix(T_inf_expanded)

    N = int(np.log2(T_inf_expanded.shape[0]))

    state_prob = np.mean(T_inf_expanded, axis=0)

    node_average_values = np.zeros(N)

    for i, prob in enumerate(state_prob):
        if prob != 0:
            # convert i into binary string
            state_str = bin(i)[2:].zfill(N)
            state_value = np.array([float(state_str[j]) for j in range(N)])
            contribution = prob * state_value
            node_average_values += contribution

    # if any value is greater than 1, set it to 1
    node_average_values[node_average_values > 1] = 1

    # turn it into a (1, N) array
    node_average_values = np.expand_dims(node_average_values, axis=0)

    return node_average_values
