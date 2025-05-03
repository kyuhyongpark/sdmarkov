import numpy as np

from matrix_operations import nsquare, expand_matrix
from transition_matrix import check_transition_matrix


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
) -> dict:
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
    basin_ratios : dict
        A dictionary mapping each attractor to the ratio of the size of its basin to the total number of states.
    """

    if DEBUG:
        # Check that the given matrix is a transition matrix of size 2**N
        if T_inf.shape[0] != T_inf.shape[1]:
            raise ValueError("The matrix must be a square matrix.")
        if T_inf.shape[0] != 2 ** int(np.log2(T_inf.shape[0])):
            raise ValueError("The matrix must be a transition matrix of size 2**N.")

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

    state_prob = np.mean(T_inf, axis=0)

    attractor_ratio = {}
    for attractor in attractor_indices:
        attractor_ratio[tuple(attractor)] = state_prob[attractor].sum()

    return attractor_ratio


def get_basin_ratios_rmsd(
    answer: dict[tuple, float],
    guess: dict[tuple, float],
    DEBUG: bool = False,
) -> float:
    """
    Calculate the root mean squared difference between the basin ratios in two dictionaries.

    Parameters
    ----------
    answer : dict[tuple, float]
        The ground truth basin ratios.
    guess : dict[tuple, float]
        The predicted basin ratios.
    DEBUG : bool, optional
        If True, performs additional checks.

    Returns
    -------
    rmsd : float
        The root mean squared difference between the two dictionaries.
    """
    if DEBUG:
        # Check that the keys are the same
        if set(answer.keys()) != set(guess.keys()):
            raise ValueError("The keys must be the same.")
        
        # Check that the keys are in the same order
        if list(answer.keys()) != list(guess.keys()):
            # If the keys are not in the same order, sort them
            answer = {k: answer[k] for k in sorted(answer)}
            guess = {k: guess[k] for k in sorted(guess)}
    
    rmsd = np.sqrt(np.mean((np.array(list(answer.values())) - np.array(list(guess.values())))**2))

    return rmsd


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