import numpy as np
import networkx as nx

from sdmarkov.graph import get_stg, get_markov_chain
from sdmarkov.scc_dags import get_scc_dag, get_attractor_states
from sdmarkov.matrix_operations import nsquare

def get_decision_matrix(
    transition_matrix: np.ndarray,
    group_indices: list[list[int]] = None,
    attractor_indices: list[list[int]] = None,
    DEBUG: bool = False,
) -> np.ndarray:
    """
    This function returns a matrix with the same shape as the transition matrix, 
    where each entry encodes whether the corresponding transition is a decision.

    Parameters
    ----------
    transition_matrix : numpy array
        The transition matrix of a Markov chain. The entry at row i and column j is the probability of transitioning from state i to state j.
    group_indices : list
        A list of lists containing the indices that correspond to the states of the Boolean system in each group.
    attractor_indices : list
        A list of lists containing the indices that correspond to the states of the Markov chain in each attractor.
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
    if attractor_indices == None:
        if group_indices == None:
            stg = get_stg(transition_matrix, DEBUG=DEBUG)
            scc_dag = get_scc_dag(stg)
            attractor_indices = get_attractor_states(scc_dag, as_indices=True, DEBUG=DEBUG)
        else:
            markov_chain = get_markov_chain(transition_matrix, group_indices, DEBUG=DEBUG)
            scc_dag = get_scc_dag(markov_chain)
            attractor_indices = get_attractor_states(scc_dag, as_indices=True, as_groups=True, DEBUG=DEBUG)

    if DEBUG:
        # check if the attractor indices are between 0 and transition_matrix.shape[0]
        for attractor in attractor_indices:
            for index in attractor:
                if index < 0 or index >= transition_matrix.shape[0]:
                    raise ValueError(f"attractor indices must be between 0 and transition_matrix.shape[0] but got {index}")

        # check if the attractor indices are unique
        for attractor in attractor_indices:
            if len(attractor) != len(set(attractor)):
                raise ValueError("attractor indices must be unique")

    # get the number of reachable attractors
    T_inf = nsquare(transition_matrix, 20, DEBUG=DEBUG)

    n_reachable_attractors = np.zeros((transition_matrix.shape[0], transition_matrix.shape[1]))
    for row in range(transition_matrix.shape[0]):
        reachable = 0
        for attractor in attractor_indices:
            for index in attractor:
                if T_inf[row, index] != 0:
                    reachable += 1
                    break
        n_reachable_attractors[row, :] = reachable
    
    if DEBUG:
        if np.any(n_reachable_attractors==0):
            raise ValueError("a state must go into at least one attractor")

    # get the basin relation
    transpose = np.transpose(n_reachable_attractors)
    n_reachable_relation = n_reachable_attractors - transpose

    # get the decision matrix
    # Define the conditions and corresponding values
    conditions = [
        (transition_matrix != 0) & (n_reachable_relation > 0),  # decision
        (transition_matrix != 0) & (n_reachable_relation == 0), # no decision
        (transition_matrix != 0) & (n_reachable_relation < 0),  # impossible
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
    decision_matrix: np.ndarray,
    group_indices: list[list[int]],
    DEBUG: bool = False
) -> np.ndarray:
    """
    Expand a compressed decision matrix by splitting certain rows and columns into multiple rows and columns.

    Parameters
    ----------
    decision_matrix : np.ndarray
        The compressed matrix to expand.
    group_indices : list[list[int]]
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
    >>> group_indices = [[0, 1], [2]]
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

        # Check if there are any empty index groups
        if any(len(group) == 0 for group in group_indices):
            raise ValueError("Index groups cannot be empty.")

        # Check that index groups are mutually exclusive
        for i in range(len(group_indices)):
            for j in range(i + 1, len(group_indices)):
                if set(group_indices[i]).intersection(set(group_indices[j])):
                    raise ValueError("Index groups must be mutually exclusive.")

        # Check that the number of index groups is not greater than the size of the matrix
        if len(group_indices) != decision_matrix.shape[0]:
            raise ValueError(
                "The number of index groups must be less equal to the size of the matrix."
            )

    compressed_matrix_dimension = len(decision_matrix)

    all_indices = []
    for group in group_indices:
        all_indices.extend(group)

    expanded_matrix_dimension = (
        compressed_matrix_dimension - len(group_indices) + len(all_indices)
    )

    row_expanded = np.zeros((expanded_matrix_dimension, compressed_matrix_dimension))

    j = 0
    for i in range(expanded_matrix_dimension):
        if i not in all_indices:
            raise ValueError("Missing index.")
        for k, group in enumerate(group_indices):
            if i in group:
                row_expanded[i] = decision_matrix[k]
                break

    expanded = np.zeros((expanded_matrix_dimension, expanded_matrix_dimension))

    j = 0
    for i in range(expanded_matrix_dimension):
        for k, group in enumerate(group_indices):
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
