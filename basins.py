import numpy as np

from transition_matrix import check_transition_matrix
from helper import indices_to_states


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


# def get_node_average_values(
#     transition_matrix: np.ndarray,
#     attractor_indices: list[list[int]] = None,
#     scc_dag: nx.DiGraph = None,
#     stg: nx.DiGraph = None,
#     DEBUG: bool = False
# ):

#     # start with getting the stg
#     if attractor_indices == None and scc_dag == None and stg == None:
#         stg = get_stg(transition_matrix, DEBUG=DEBUG)

#     # start with getting the scc dag
#     if attractor_indices == None and scc_dag == None and stg != None:
#         scc_dag = get_scc_dag(stg)

#     # start with getting the attractor index
#     if attractor_indices == None and scc_dag != None:
#         attractor_indices = get_attractor_states(scc_dag, as_indices=True, DEBUG=DEBUG)

#     # get the basins
#     T_inf = nsquare(transition_matrix, 20, DEBUG=DEBUG)

#     state_prob = np.mean(T_inf, axis=0)

#     all_attractor_states = []
#     for attractor in attractor_indices:
#         all_attractor_states.extend(attractor)

#     print(all_attractor_states)

#     node_average_values = np.zeros(len(primes))
#     for state in all_attractor_states:

#         # convert decimal index to binary string
#         state_str = bin(state)[2:].zfill(len(primes))

#         node_values = np.array([float(state_str[i]) for i in range(len(state_str))])

#         print(f"Node values for state {state}: {node_values}")

#         node_average_values += node_values * state_prob[state]