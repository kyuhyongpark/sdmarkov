import numpy as np
import networkx as nx

from transition_matrix import get_stg
from scc_dags import get_scc_dag, get_attractor_states
from matrix_operations import nsquare

def get_strong_basins(
    transition_matrix: np.ndarray, 
    attractor_indexes: list[list[int]] = None, 
    scc_dag: nx.DiGraph = None, 
    stg: nx.DiGraph = None, 
    DEBUG: bool = False,
) -> np.ndarray:
    """
    Get the strong basins of a Boolean network.

    Parameters
    ----------
    transition_matrix : numpy array
        The transition matrix.
    attractor_indexes : list[list[int]], optional
        The indices of the attractor states in the transition matrix.
        Note that this should match the indices in the given transition matrix.
    scc_dag : networkx DiGraph, optional
        The SCC DAG.
    stg : networkx DiGraph, optional
        The state transition graph.
    DEBUG : bool, optional
        If set to True, performs additional checks.

    Returns
    -------
    strong_basin : numpy array
        The strong basin for each state in the transition matrix.
    """

    # start with getting the stg
    if attractor_indexes == None and scc_dag == None and stg == None:
        stg = get_stg(transition_matrix, DEBUG=DEBUG)

    # start with getting the scc dag
    if attractor_indexes == None and scc_dag == None and stg != None:
        scc_dag = get_scc_dag(stg)

    # start with getting the attractor index
    if attractor_indexes == None and scc_dag != None:
        attractor_indexes = get_attractor_states(scc_dag, as_indexes=True, DEBUG=DEBUG)

    # get the basins
    T_inf = nsquare(transition_matrix, 20, DEBUG=DEBUG)

    strong_basin = np.zeros((transition_matrix.shape[0], 1))
    for row in range(transition_matrix.shape[0]):
        single = False
        multiple = False
        for i, attractor in enumerate(attractor_indexes):
            for index in attractor:
                # the attractor can be reached
                if T_inf[row, index] != 0:
                    if single:
                        multiple = True
                    else:
                        single = True
                        attractor_index = i
                    break
        
        if single and not multiple:
            strong_basin[row] = attractor_index
        else:
            strong_basin[row] = -1
        
        if DEBUG and not single and not multiple:
            raise ValueError(f"Must have at least one attractor for row {row}")

    return strong_basin


def expand_strong_basin_matrix(
    strong_basin: np.ndarray, index_groups: list[list[int]], DEBUG: bool = False
) -> np.ndarray:
    """
    Expand a compressed strong basin matrix by repeating certain rows into multiple rows.

    Parameters
    ----------
    strong_basin : np.ndarray
        The compressed matrix to expand.
    index_groups : list[list[int]]
        A list of index groups, where each group specifies the rows and columns
        to be merged into a single row and column.
    DEBUG : bool, optional
        If True, perform basic checks.

    Returns
    -------
    numpy.ndarray
        The expanded matrix.

    """
    if DEBUG:
        # Check that index groups are mutually exclusive
        for i in range(len(index_groups)):
            for j in range(i + 1, len(index_groups)):
                if set(index_groups[i]).intersection(set(index_groups[j])):
                    raise ValueError("Index groups must be mutually exclusive.")

        # Check that the number of non-empty index groups is not greater than the size of the matrix
        non_empty_groups = [group for group in index_groups if group]
        if len(non_empty_groups) > strong_basin.shape[0]:
            raise ValueError(
                "The number of non-empty index groups must be less than or equal to the size of the matrix."
            )

    compressed_matrix_dimension = len(strong_basin)

    all_indexes = []
    for group in index_groups:
        all_indexes.extend(group)

    non_empty_groups = [group for group in index_groups if group]

    expanded_matrix_dimension = (
        compressed_matrix_dimension - len(non_empty_groups) + len(all_indexes)
    )

    expanded = np.zeros((expanded_matrix_dimension, 1))

    j = 0
    for i in range(expanded_matrix_dimension):
        if i not in all_indexes:
            # First len(non_empty_groups) rows of the matrix are the compressed rows,
            # and the rest are the not-compressed rows.
            # Here restore the j-th not-compressed row.
            expanded[i] = strong_basin[len(non_empty_groups) + j]
            j += 1
        else:
            for k, group in enumerate(non_empty_groups):
                if i in group:
                    expanded[i] = strong_basin[k]
                    break

    return expanded


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
    TP = np.sum((guess != -1) & (answer == guess))  # True positives
    FP = np.sum((guess != -1) & (answer != guess))  # False positives
    TN = np.sum((guess == -1) & (answer == guess))  # True negatives
    FN = np.sum((guess == -1) & (answer != guess))  # False negatives

    if DEBUG:
        if FP > 0:
            raise ValueError("Markov chain should not have false positives.")

    return TP, FP, TN, FN

def get_basin_ratios(
    T_inf: np.ndarray,
    attractor_indexes: list[list[int]],
    DEBUG: bool = False,
) -> dict:
    """
    Calculate the ratio of the size of each basin to the total number of states.

    Parameters
    ----------
    T_inf : numpy array
        The transition matrix at t=inf.
        Note that this should be 2**N x 2**N.
    attractor_indexes : list[list[int]], optional
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

        # Check that the attractor indexes are valid
        for attractor in attractor_indexes:
            for state in attractor:
                if 0 > state or state >= T_inf.shape[0]:
                    raise ValueError("The attractor indexes must be valid.")

        # Check that the attractor indexes are mutually exclusive
        for i in range(len(attractor_indexes)):
            for j in range(i + 1, len(attractor_indexes)):
                if set(attractor_indexes[i]).intersection(set(attractor_indexes[j])):
                    raise ValueError("The attractor indexes must be mutually exclusive.")

    state_prob = np.mean(T_inf, axis=0)

    attractor_ratio = {}
    for attractor in attractor_indexes:
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
#     attractor_indexes: list[list[int]] = None,
#     scc_dag: nx.DiGraph = None,
#     stg: nx.DiGraph = None,
#     DEBUG: bool = False
# ):

#     # start with getting the stg
#     if attractor_indexes == None and scc_dag == None and stg == None:
#         stg = get_stg(transition_matrix, DEBUG=DEBUG)

#     # start with getting the scc dag
#     if attractor_indexes == None and scc_dag == None and stg != None:
#         scc_dag = get_scc_dag(stg)

#     # start with getting the attractor index
#     if attractor_indexes == None and scc_dag != None:
#         attractor_indexes = get_attractor_states(scc_dag, as_indexes=True, DEBUG=DEBUG)

#     # get the basins
#     T_inf = nsquare(transition_matrix, 20, DEBUG=DEBUG)

#     state_prob = np.mean(T_inf, axis=0)

#     all_attractor_states = []
#     for attractor in attractor_indexes:
#         all_attractor_states.extend(attractor)

#     print(all_attractor_states)

#     node_average_values = np.zeros(len(primes))
#     for state in all_attractor_states:

#         # convert decimal index to binary string
#         state_str = bin(state)[2:].zfill(len(primes))

#         node_values = np.array([float(state_str[i]) for i in range(len(state_str))])

#         print(f"Node values for state {state}: {node_values}")

#         node_average_values += node_values * state_prob[state]