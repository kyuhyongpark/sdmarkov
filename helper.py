import networkx as nx
import numpy as np


def states_to_indices(state_groups: list[list[str]], DEBUG: bool = False) -> list[list[int]]:
    """
    Convert groups of binary strings to groups of indices.

    Parameters
    ----------
    state_groups: list[list[str]]
        A list of sublists, where each sublist contains binary strings that represent states.
    DEBUG: bool, optional
        If True, checks if all state strings have the same length and are mutually exclusive.

    Returns
    -------
    list[list[int]]
        A list of sublists, where each sublist contains indices that correspond to the states.
    
    Examples
    --------
    >>> states_to_indices([['0100', '0101', '0110', '0111', '1101', '1111'], ['1001', '1011'], ['0000', '0001', '0010'], ['0011'], ['1100', '1110'], ['1000', '1010']])
    [[4, 5, 6, 7, 13, 15], [9, 11], [0, 1, 2], [3], [12, 14], [8, 10]]
    """

    if DEBUG:
        # Check if all state strings are of the same length
        lengths = [len(state) for state_group in state_groups for state in state_group]
        if len(set(lengths)) != 1:
            raise ValueError("Not all states have the same length")
        
        # Check if all groups are mutually exclusive
        all_states = [state for state_group in state_groups for state in state_group]
        if len(all_states) != len(set(all_states)):
            raise ValueError("States are not mutually exclusive (duplicates found)")

    # Convert binary strings to integers
    index_groups = [[int(state, 2) for state in state_group] for state_group in state_groups]
    
    return index_groups


def indices_to_states(index_groups: list[list[int]], N: int, DEBUG: bool = False) -> list[list[str]]:
    """
    Convert groups of indices to groups of states.

    Parameters
    ----------
    index_groups: list[list[int]]
        A list of sublists, where each sublist contains indices that correspond to the states.
    N: int
        The number of nodes in a state.
    DEBUG: bool, optional
        If True, checks if all groups are mutually exclusive.

    Returns
    -------
    list[list[str]]
        A list of sublists, where each sublist contains binary strings that correspond to the indices.
    
    Examples
    --------
    >>> indices_to_states([[4, 5, 6, 7, 13, 15], [9, 11], [0, 1, 2], [3], [12, 14], [8, 10]], 4)
    [['0100', '0101', '0110', '0111', '1101', '1111'], ['1001', '1011'], ['0000', '0001', '0010'], ['0011'], ['1100', '1110'], ['1000', '1010']]
    """

    if DEBUG:
        # Check if N is an integer
        if not isinstance(N, int):
            raise ValueError(f"N must be an integer: {N=}")

        # Check if N is large enough
        largest_index = max([max(index_group) for index_group in index_groups if index_group])
        if largest_index >= 2**N:
            raise ValueError("N is too small")

        # Check if all groups are mutually exclusive
        all_indices = [index for index_group in index_groups for index in index_group]
        if len(all_indices) != len(set(all_indices)):
            raise ValueError("Indices are not mutually exclusive (duplicates found)")

    # Convert integers to binary strings
    state_groups = [[f"{index:0{N}b}" for index in index_group] for index_group in index_groups]
    
    return state_groups


def check_transition_matrix(transition_matrix: np.ndarray, compressed: bool=False, partial: bool=False) -> None:
    """
    Validate the structure and properties of a transition matrix.

    Parameters
    ----------
    transition_matrix : np.ndarray, shape (2^N, 2^N)
        The matrix representing state transitions.
    compressed : bool, optional
        If True, the matrix is not required to have dimensions of 2^N.

    Raises
    ------
    ValueError
        If the matrix does not meet the specified criteria.

    Notes
    -----
    - The matrix should be square.
    - All elements must be between 0 and 1.
    - Each row must sum to 1.
    - If `compressed` is False, the number of rows/columns should be 2^N.
    """

    # Check if all elements of the matrix are numeric
    if not np.issubdtype(transition_matrix.dtype, np.number):
        raise ValueError("All elements of the matrix must be numeric.")

    # Check that the elements of the array are between 0 and 1, with some tolerance
    if not np.all(transition_matrix >= 0 - 1e-16) or not np.all(transition_matrix <= 1 + 1e-16):
        raise ValueError("All elements of the matrix must be between 0 and 1. Max: {}, Min: {}".format(np.max(transition_matrix), np.min(transition_matrix)))

    if not partial:
        # Check that the matrix is square
        if transition_matrix.shape[0] != transition_matrix.shape[1]:
            raise ValueError("The matrix must be square.")

        # Check that every row of the matrix sums to 1
        if not np.allclose(np.sum(transition_matrix, axis=1), np.ones(transition_matrix.shape[1])):
            raise ValueError("Every row of the matrix must sum to 1.")

    if not compressed and not partial:
        # Check if the length of the matrix is 2^N
        N = int(np.log2(transition_matrix.shape[0]))
        if 2**N != transition_matrix.shape[0]:
            raise ValueError("The length of the matrix must be 2^N.")


def check_stg(stg: nx.DiGraph) -> None:
    """
    Check if a given state transition graph is valid.
    TODO: apply different checks for different update schemes
    
    A state transition graph must satisfy the following conditions:

    1. The number of nodes in the state transition graph is 2^N, where N is the number of nodes in a state;
    2. Each node in the state transition graph must be a string of 0s and 1s;
    3. All states in the state transition graph have the same length of N;
    4. N should be a positive integer.
    5. The number of outgoing edges for each state is less than or equal to N;

    If any of these conditions are not met, a ValueError is raised.

    Parameters
    ----------
    stg : networkx DiGraph
        The state transition graph to check.
    """
    
    # Check if each state is a string of 0s and 1s
    for state in stg.nodes():
        if not isinstance(state, str):
            raise ValueError("Each state in the state transition graph must be a string of 0s and 1s.")
        if not all(char in ['0', '1'] for char in state):
            raise ValueError("Each state in the state transition graph must be a string of 0s and 1s.")

    # Get the number of nodes in a state
    N = len(list(stg.nodes())[0])

    # N should be a positive integer
    if N <= 0:
        raise ValueError("N should be a positive integer.")
    
    # Check if all states have the same length of N
    lengths = [len(node) for node in stg.nodes()]
    if set(lengths) != set([N]):
        raise ValueError("All states in the state transition graph must have the same length of N.")

    # Check if the number of states in the state transition graph is 2^N
    if 2**N != stg.number_of_nodes():
        raise ValueError("The number of states in the state transition graph must be 2^N.")
    
    # Check if the number of outgoing transitions for each state is less than or equal to N
    for state in stg.nodes():
        if stg.out_degree(state) > N:
            raise ValueError("The number of outgoing transitions for each state must be less than or equal to N.")