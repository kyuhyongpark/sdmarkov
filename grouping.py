import math
import random
from collections import Counter

import matplotlib.pyplot as plt

from succession_diagram import get_sd_nodes_and_edges, get_sd_group_states
from helper import states_to_indices


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
    [[4, 5, 6, 7], [], [0, 1, 2], [3], [12, 14], [8, 10], [13, 15], [9, 11]]
    """
    nodes, sd_nodes, sd_edges = get_sd_nodes_and_edges(bnet)

    success, sd_group_states, duplicates = get_sd_group_states(nodes, sd_nodes, sd_edges, DEBUG=DEBUG)

    if DEBUG:
        if duplicates:
            print(f"{duplicates=}")
    
    extra_groups = []
    n_trys = 0
    while not success and n_trys < 100:
        n_trys += 1
        
        for state in duplicates:
            extra_group = {}
            for node_or_edge in duplicates[state]:
                if isinstance(node_or_edge, dict): # node or extra group
                    extra_group.update(node_or_edge)
                else: # edge
                    for motif in node_or_edge:
                        # Check whether the state agrees with the motif
                        agree = True
                        for i, node in enumerate(nodes):
                            if node not in motif:
                                continue
                            if int(state[i]) != motif[node]:
                                agree = False
                                break
                        if agree:
                            extra_group.update(motif)
            extra_group = dict(sorted(extra_group.items(), key=lambda x: x[0]))

            if extra_group not in extra_groups:
                extra_groups.append(extra_group)

        success, sd_group_states, duplicates = get_sd_group_states(nodes, sd_nodes, sd_edges, extra_groups, DEBUG=DEBUG)

    if DEBUG:
        if extra_groups:
            print(f"{extra_groups=}")

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
    nodes, min_trap_nodes, _ = get_sd_nodes_and_edges(bnet, minimal=True)

    # Make sure to add the group for all transient states
    if {} not in min_trap_nodes:
        min_trap_nodes.insert(0, {})

    success, min_trap_states, duplicates = get_sd_group_states(nodes, min_trap_nodes, sd_edges=_, DEBUG=DEBUG)

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


# Function to calculate the number of configurations based on the sublist lengths
def calculate_combinations(lengths):
    total_length = sum(lengths)
    result = math.factorial(total_length)
    
    # order within each sublist is irrelevant
    for length in lengths:
        result //= math.factorial(length)
    
    # sublists of the same length are indistinguishable
    length_distributions = Counter(lengths)
    for length, count in length_distributions.items():
        result //= math.factorial(count)

    return result


# Function to run and compare the results with possible combinations
def compare_with_possible_combinations(input_list, N, m, num_runs=1000):
    # Store the length combinations
    length_combinations = []
    
    for i in range(num_runs):
        sublists = divide_list_into_sublists(input_list, N, m, seed=i)
        sublist_lengths = [len(sublist) for sublist in sublists]
        length_combinations.append(tuple(sorted(sublist_lengths)))
    
    # Count the frequencies of length combinations
    length_frequencies = Counter(length_combinations)
    
    # Calculate the number of configurations for each unique combination
    combinations_with_configurations = {}

    for combination, frequency in length_frequencies.items():
        configurations = calculate_combinations(combination)
        combinations_with_configurations[combination] = {'frequency': frequency, 'configurations': configurations}

    # sort the combinations_with_configurations by combination
    combinations_with_configurations = dict(sorted(combinations_with_configurations.items()))

    # Output comparison results
    for combination, data in combinations_with_configurations.items():
        print(f"  {combination}: Frequency = {data['frequency']}, Possible configurations = {data['configurations']}, ratio = {data['frequency'] / data['configurations']}")
    
    # Plot histogram comparing the distributions of sublist lengths for each method   
    plt.figure(figsize=(10, 6))

    flattened_lengths = [length for lengths in length_combinations for length in lengths]
    plt.hist(flattened_lengths, bins=range(1, max(flattened_lengths) + 2), edgecolor='black', alpha=0.5)
    
    plt.title('Distribution of Sublist Lengths')
    plt.xlabel('Sublists Length')
    plt.ylabel('Frequency')
    plt.xticks(range(1, max(flattened_lengths) + 1))
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Example input and method parameters
    input_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    N = 3  # Number of sublists
    m = 1  # Minimum number of elements in each sublist
    num_runs = 100000  # Number of simulations

    # Run the comparison
    compare_with_possible_combinations(input_list, N, m, num_runs)