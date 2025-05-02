import numpy as np

from matrix_operations import nsquare, expand_matrix
from transition_matrix import check_transition_matrix


def get_strong_basins(
    transition_matrix: np.ndarray,
    attractor_indices: list[list[int]],
    grouped: bool = False,
    group_indices: list[list[int]] = None,
    exclude_attractors: bool = False,
    DEBUG: bool = False,
) -> np.ndarray:
    """
    Compute the strong basin of a transition matrix.

    Parameters
    ----------
    transition_matrix : numpy array
        The transition matrix of the Boolean network.
    attractor_indices : list[list[int]]
        The indices of the attractor states in the transition matrix.
        The order of the attractors is used to assign the attractor number.
    grouped : bool, optional
        If True, the transition matrix is grouped.
    group_indices : list[list[int]], optional
        The group indices of the transition matrix.
        If not given, the transition matrix is not grouped.
    exclude_attractors : bool, optional
        If True, the attractor states are excluded from the strong basin.
    DEBUG : bool, optional
        If True, performs additional checks on the input data.

    Returns
    -------
    strong_basins : numpy array, shape (2^N, 1)
        The strong basin of each state in the transition matrix.
        The value at each row is the attractor number of the strong basin that the state is in.
        If the state is not in a strong basin, the value is -1.
        If the state is part of an attractor, the value is -2.
    """
    if DEBUG:
        if grouped and group_indices == None:
            raise ValueError("If grouped is True, group_indices must be given.")

    # get the basins
    T_inf = nsquare(transition_matrix, 20, DEBUG=DEBUG)

    if grouped:
        T_inf = expand_matrix(T_inf, group_indices, DEBUG=DEBUG)

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

    if exclude_attractors:
        all_attractor_states = []
        for attractor in attractor_indices:
            all_attractor_states.extend(attractor)

    strong_basin = np.zeros((T_inf.shape[0], 1))
    for row in range(T_inf.shape[0]):

        if exclude_attractors and row in all_attractor_states:
            strong_basin[row] = -2
            continue

        single = False
        multiple = False
        # iterate through each attractor
        for i, attractor in enumerate(attractor_indices):
            for state in attractor:
                # the attractor can be reached
                if T_inf[row, state] != 0:
                    # This is the first attractor that can be reached
                    if not single:
                        single = True
                        attractor_index = i
                    # This is not the first attractor that can be reached
                    else:
                        single = False
                        multiple = True
                        attractor_index = -1
                    break
            if multiple:
                break
        strong_basin[row] = attractor_index
        
        if DEBUG:
            if not single and not multiple:
                raise ValueError(f"Must have at least one attractor for row {row}")
            elif single and multiple:
                raise ValueError(f"Row {row} has both a single attractor and multiple attractors")

    return strong_basin


def compare_strong_basins(
    answer: np.ndarray, 
    guess: np.ndarray, 
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
    DEBUG : bool, optional
        If True, perform basic checks.

    Returns
    -------
    tuple of int
        A tuple containing four integers: (TP, FP, TN, FN)
        - TP: Number of true positives
        - FP: Number of false positives
        - TN: Number of true negatives
        - FN: Number of false negatives
    """

    if DEBUG:
        # Check that the matrices have the same shape
        if answer.shape != guess.shape:
            raise ValueError("The matrices must have the same shape.")

    # Define the conditions for each category
    TP = np.sum((answer != -2) & (guess != -1) & (answer == guess))  # True positives
    FP = np.sum((answer != -2) & (guess != -1) & (answer != guess))  # False positives
    TN = np.sum((answer != -2) & (guess == -1) & (answer == guess))  # True negatives
    FN = np.sum((answer != -2) & (guess == -1) & (answer != guess))  # False negatives

    if DEBUG:
        if FP > 0:
            raise ValueError("Markov chain should not have false positives.")

    return TP, FP, TN, FN

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