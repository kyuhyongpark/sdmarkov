import random

from sdmarkov.representation import states_to_indices
from sdmarkov.succession_diagram import build_sd_trap_spaces, assign_states_to_trap_spaces


def sd_grouping(bnet: str, DEBUG: bool = False) -> list[list[int]]:
    """
    Get a sd grouping of a state index by when given a Boolean network in bnet format.

    Parameters
    ----------
    bnet : str
        The Boolean network as a string, with nodes and their update rules
        separated by commas.
    DEBUG : bool, optional
        If True, performs additional checks.

    Returns
    -------
    list[list[int]]
        A list of sorted integers, where each sublist corresponds to the indices to be grouped.
    
    Examples
    --------
    >>> sd_grouping("A, A | B & C\nB, B & !C\nC, B & !C | !C & !D | !B & C & D\nD, !A & !B & !C & !D | !A & C & D")
    [[4, 5, 6, 7], [0, 1, 2], [3], [13, 15], [12, 14], [9, 11], [8, 10]]
    """
    nodes, trap_spaces = build_sd_trap_spaces(bnet, DEBUG=DEBUG)
    sd_group_states = assign_states_to_trap_spaces(nodes, trap_spaces, DEBUG=DEBUG)

    indices = states_to_indices(sd_group_states, DEBUG=DEBUG)

    # remove empty groups
    indices = [group for group in indices if group]

    return indices

def null_grouping(bnet: str, DEBUG: bool = False) -> list[list[int]]:
    """
    Get a null grouping of a state index by when given a Boolean network in bnet format.
    Minimal trapspaces are grouped together, and all other states are grouped together.

    Parameters
    ----------
    bnet : str
        The Boolean network as a string, with nodes and their update rules
        separated by commas.
    DEBUG : bool, optional
        If True, performs additional checks.

    Returns
    -------
    list[list[int]]
        A list of sorted integers, where each sublist corresponds to the indices to be grouped.
    
    Examples
    --------
    >>> null_grouping("A, A | B & C\nB, B & !C\nC, B & !C | !C & !D | !B & C & D\nD, !A & !B & !C & !D | !A & C & D")
    [[0, 1, 2, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15], [3], [8, 10]]
    """
    nodes, min_trap_nodes = build_sd_trap_spaces(bnet, minimal=True, DEBUG=DEBUG)

    # Make sure to add the group for all transient states
    if {} not in min_trap_nodes:
        min_trap_nodes.insert(0, {})

    min_trap_states = assign_states_to_trap_spaces(nodes, min_trap_nodes, DEBUG=DEBUG)
    indices = states_to_indices(min_trap_states, DEBUG=DEBUG)

    # remove empty groups
    indices = [group for group in indices if group]

    return indices

def random_grouping(
    sd_indices: list[list[int]],
    null_indices: list[list[int]],
    smallest_group_size: int = 1,
    seed: int|None = None,
    DEBUG: bool = False,
) -> list[list[int]]:
    """
    Divide the transient states into random groups.

    Parameters
    ----------
    sd_indices : list[list[int]]
        The groups of indices of the states that correspond to succession diagram nodes.
    null_indices : list[list[int]]
        The groups of indices of the states in the minimal trapspaces and the transient states.
    smallest_group_size : int, optional
        The smallest number of elements in each group. Defaults to 1.
    seed : int|None, optional
        The random seed to use. Defaults to None.
    DEBUG : bool, optional
        If True, performs additional checks. Defaults to False.

    Returns
    -------
    list[list[int]]
        A list of lists of indices, where each sublist is a group of transient states.
    """
    if DEBUG:
        # All index groups in null_indices except the first one should be in sd_indices
        for index_group in null_indices[1:]:
            if index_group not in sd_indices:
                raise ValueError(f"{index_group} is in null_indices but not in sd_indices.")
            
        # All indices of sd_indices and null_indices should be unique
        all_sd_indices = []
        for index_group in sd_indices:
            all_sd_indices.extend(index_group)
        all_null_indices = []
        for index_group in null_indices:
            all_null_indices.extend(index_group)
        if len(set(all_sd_indices)) != len(all_sd_indices) or len(set(all_null_indices)) != len(all_null_indices):
            raise ValueError("All indices should be unique in sd_indices and null_indices.")
        
        # All indices should be present in both sd_indices and null_indices
        all_indices = []
        for index_group in sd_indices:
            all_indices.extend(index_group)
        for index_group in null_indices:
            all_indices.extend(index_group)
        if len(set(all_indices)) != len(all_sd_indices) or len(set(all_indices)) != len(all_null_indices):
            raise ValueError("All indices should be present in both sd_indices and null_indices.")
    
    # Get the indices of the transient states from the null_indices
    transient_indices = null_indices[0]

    # if transient_indices is empty, return null_indices
    if not transient_indices:
        return null_indices

    # Get the number of non-empty groups in sd_indices and null_indices
    non_empty_sd_indices = len([index_group for index_group in sd_indices if index_group])
    non_empty_null_indices = len([index_group for index_group in null_indices if index_group])

    # Get the number of groups
    num_groups = non_empty_sd_indices - non_empty_null_indices + 1

    # Divide the transient states into num_groups
    indices = divide_list_into_sublists(transient_indices, num_groups, smallest_group_size, seed=seed)

    indices.extend(null_indices[1:])

    # remove empty groups
    indices = [group for group in indices if group]

    return indices


def divide_list_into_sublists(
    lst: list,
    N: int,
    m: int,
    seed: int|None = None,
) -> list[list]:
    """
    Divide a list into N sublists of at least m elements each.

    Parameters
    ----------
    lst : list
        The list to be divided.
    N : int
        The number of sublists to divide the list into.
    m : int
        The minimum size of each sublist.
    seed : int|None, optional
        The random seed to use.

    Returns
    -------
    list[list]
        A list of N sublists, each with at least m elements.

    Examples
    --------
    >>> divide_list_into_sublists([1, 2, 3, 4, 5], 2, 2)
    [[1, 3, 4], [2, 5]]
    """
    # Set the random seed
    if seed is not None:
        random.seed(seed)

    # Check if N is valid
    if N < 1 or N > len(lst):
        raise ValueError(f"N must be between 1 and the length of the list ({len(lst)}), inclusive.")
    
    # Check if m is valid
    if m < 1:
        raise ValueError("m must be at least 1.")
    
    # Check that N * m does not exceed the length of the list
    if N * m > len(lst):
        raise ValueError(f"Not enough elements in the list to create {N} sublists with at least {m} elements each.")

    while True:
        # Initialize N empty sublists
        sublists = [[] for _ in range(N)]
        
        # Randomly assign each element to one of the sublists
        for item in lst:
            random_index = random.randint(0, N-1)
            sublists[random_index].append(item)
        
        # Check if all sublists have at least m elements
        if all(len(sublist) >= m for sublist in sublists):  # If no sublist is smaller than m
            return sorted(sublists)

