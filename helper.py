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
        largest_index = max([max(index_group) for index_group in index_groups])
        if largest_index >= 2**N:
            raise ValueError("N is too small")

        # Check if all groups are mutually exclusive
        all_indices = [index for index_group in index_groups for index in index_group]
        if len(all_indices) != len(set(all_indices)):
            raise ValueError("Indices are not mutually exclusive (duplicates found)")

    # Convert integers to binary strings
    state_groups = [[f"{index:0{N}b}" for index in index_group] for index_group in index_groups]
    
    return state_groups