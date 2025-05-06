import numpy as np

from transition_matrix import check_transition_matrix
from helper import indices_to_states


def get_strong_basins(
    convergence_matrix: np.ndarray,
    DEBUG: bool = False,
) -> np.ndarray:
    """
    Compute whether the transient states belong to a strong basin.

    Parameters
    ----------
    convergence_matrix : numpy array
        A matrix that represents the probability of reaching each attractor from each transient state
    DEBUG : bool, optional
        If True, performs additional checks on the input data.

    Returns
    -------
    strong_basins : numpy array, shape (number of transient states, 1)
        If the state is in a strong basin, the value is 1.
        If the state is in a weak basin instead, the value is 0.

    Notes
    -----
    The order of the transient states follow the order of states in the convergence matrix.
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
    T_inf: np.ndarray,
    attractor_indices: list[list[int]],
    DEBUG: bool = False,
) -> tuple[np.ndarray, list[list[str]]]:
    """
    Calculate the ratio of the size of each basin to the total number of states.

    Parameters
    ----------
    T_inf : numpy array
        The transition matrix at t=inf.
        Note that this should be 2**N x 2**N.
    attractor_indices : list[list[int]], optional
        The indices of the attractor states in the transition matrix.
    DEBUG : bool, optional
        If True, performs additional checks.

    Returns
    -------
    attractor_ratio : 2D numpy array
        The probability of reaching each attractor.
        Note that it is 2D for compatibility with other functions.
        Each row must sum to 1.
    attractor_states : list[list[str]]
        The states of each attractor.
    """

    if DEBUG:
        check_transition_matrix(T_inf)

        # Check that the attractor indices are valid
        for attractor in attractor_indices:
            for state in attractor:
                if 0 > state or state >= T_inf.shape[0]:
                    raise ValueError("The attractor indices must be valid.")

        # Check that the attractor indices are mutually exclusive
        for i in range(len(attractor_indices)):
            for j in range(i + 1, len(attractor_indices)):
                if set(attractor_indices[i]).intersection(set(attractor_indices[j])):
                    raise ValueError("The attractor indices must be mutually exclusive.")

    N = int(np.log2(T_inf.shape[0]))

    # Get the attractor states
    attractor_states = indices_to_states(attractor_indices, N, DEBUG=DEBUG)

    state_prob = np.mean(T_inf, axis=0)

    attractor_ratio = np.zeros((1, len(attractor_indices)))
    for i, attractor in enumerate(attractor_indices):
        attractor_ratio[0][i] = state_prob[attractor].sum()

    # Normalize the basin ratios
    attractor_ratio /= attractor_ratio.sum()

    return attractor_ratio, attractor_states


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