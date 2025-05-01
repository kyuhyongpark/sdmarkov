import numpy as np
import networkx as nx

from transition_matrix import get_stg
from scc_dags import get_scc_dag, get_attractor_states
from matrix_operations import nsquare

def get_decision_matrix(
    transition_matrix: np.ndarray, 
    attractor_indices: list[list[int]] = None, 
    scc_dag: nx.DiGraph = None, 
    stg: nx.DiGraph = None, 
    DEBUG: bool = False,
) -> np.ndarray:
    """
    This function returns a matrix with the same shape as the transition matrix, 
    where each entry encodes whether the corresponding transition is a decision.

    Parameters
    ----------
    transition_matrix : numpy array
        The transition matrix. The entry at row i and column j is the probability of transitioning from state i to state j.
    attractor_indices : list[list[int]], optional
        The indices of the attractor states in the transition matrix.
    scc_dag : networkx DiGraph, optional
        The SCC DAG.
    stg : networkx DiGraph, optional
        The state transition graph.
    DEBUG : bool, optional
        If set to True, performs additional checks on the input data.

    Returns
    -------
    decision_matrix : numpy array
        The decision matrix. The entry at row i and column j is the decision for the corresponding transition.
        If the entry is 1, then the transition is a decision.
        If the entry is -1, then the transition is not a decision.
        If the entry is 0, then there is no transition.

    Notes
    -----
    basin_relation matrix : numpy array
    If state i is has more reachable attractors than state j, then basin_relation[i, j] > 0.
        If this transition exists, then it is a decision.
    If state i is has less reachable attractors than state j, then basin_relation[i, j] < 0
        Such a transition must not exist.
    If state i is has the same number of reachable attractors than state j, then basin_relation[i, j] = 0
        If this transition exists, then it is not a decision.
    """

    # start with getting the stg
    if attractor_indices == None and scc_dag == None and stg == None:
        stg = get_stg(transition_matrix, DEBUG=DEBUG)

    # start with getting the scc dag
    if attractor_indices == None and scc_dag == None and stg != None:
        scc_dag = get_scc_dag(stg)

    # start with getting the attractor index
    if attractor_indices == None and scc_dag != None:
        attractor_indices = get_attractor_states(scc_dag, as_indices=True, DEBUG=DEBUG)

    # get the basins
    T_inf = nsquare(transition_matrix, 20, DEBUG=DEBUG)

    basin = np.zeros((transition_matrix.shape[0], transition_matrix.shape[1]))
    for row in range(transition_matrix.shape[0]):
        reachable = 0
        for attractor in attractor_indices:
            for index in attractor:
                if T_inf[row, index] != 0:
                    reachable += 1
                    break
        basin[row, :] = reachable
    
    if DEBUG:
        if np.any(basin==0):
            raise ValueError("a state must go into at least one attractor")

    # get the basin relation
    basin_transpose = np.transpose(basin)
    basin_relation = basin - basin_transpose

    # get the decision matrix
    # Define the conditions and corresponding values
    conditions = [
        (transition_matrix != 0) & (basin_relation > 0),  # decision
        (transition_matrix != 0) & (basin_relation == 0), # no decision
        (transition_matrix != 0) & (basin_relation < 0),  # impossible
        (transition_matrix == 0),                         # no transition
    ]

    choices = [1, -1, 2, 0]  # Corresponding assignments for each condition

    decision_matrix = np.select(conditions, choices, default=2)
    
    if DEBUG:
        # if any element is 2, then raise an error
        if np.any(decision_matrix == 2):
            raise ValueError("Decision matrix is not valid.")
    
    return decision_matrix


def expand_decision_matrix(
    decision_matrix: np.ndarray, index_groups: list[list[int]], DEBUG: bool = False
) -> np.ndarray:
    """
    Expand a compressed decision matrix by splitting certain rows and columns into multiple rows and columns.

    Parameters
    ----------
    decision_matrix : np.ndarray
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
    >>> matrix = np.array([[-1, 1], [0, -1]])
    >>> index_groups = [[0, 1], [2]]
    >>> expanded_matrix = expand_decision_matrix(matrix, index_groups)
    >>> expanded_matrix
    array([[-1, -1,  1],
           [-1, -1,  1],
           [ 0,  0, -1]])
    """

    if DEBUG:
        # Check that decision_matrix is a 2D numpy array
        if not isinstance(decision_matrix, np.ndarray) or decision_matrix.ndim != 2:
            raise ValueError("Matrix must be a 2D numpy array.")

        # Check if the decision matrix is square
        if decision_matrix.shape[0] != decision_matrix.shape[1]:
            raise ValueError("Matrix must be square.")

        # Check that index groups are mutually exclusive
        for i in range(len(index_groups)):
            for j in range(i + 1, len(index_groups)):
                if set(index_groups[i]).intersection(set(index_groups[j])):
                    raise ValueError("Index groups must be mutually exclusive.")

        # Check that the number of non-empty index groups is not greater than the size of the matrix
        non_empty_groups = [group for group in index_groups if group]
        if len(non_empty_groups) > decision_matrix.shape[0]:
            raise ValueError(
                "The number of non-empty index groups must be less than or equal to the size of the matrix."
            )

    compressed_matrix_dimension = len(decision_matrix)

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
            row_expanded[i] = decision_matrix[len(non_empty_groups) + j]
            j += 1
        else:
            for k, group in enumerate(non_empty_groups):
                if i in group:
                    row_expanded[i] = decision_matrix[k]
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
                    expanded[:, i] = row_expanded[:, k]
                    break

    return expanded


def compare_decision_matrices(answer, guess, DEBUG=False):
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
    tuple
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
    TP = np.sum((answer == 1) & (guess == 1))  # True positives
    FP = np.sum((answer == -1) & (guess == 1))  # False positives
    TN = np.sum((answer == -1) & (guess == -1))  # True negatives
    FN = np.sum((answer == 1) & (guess == -1))  # False negatives

    if DEBUG:
        ERROR = np.sum((answer != 0) & (guess == 0))
        if ERROR > 0:
            raise ValueError("Markov chain is missing a transition.")

    return TP, FP, TN, FN
