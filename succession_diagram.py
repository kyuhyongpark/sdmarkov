from itertools import product

import biobalm

def get_sd_nodes(bnet: str, minimal: bool = False) -> tuple[list[str], list[dict[str, int]]]:
    """
    Given a Boolean network as a string in bnet format, returns a tuple containing
    a list of node names and a list of dictionaries, where each dictionary corresponds
    to the state of a node in the succession diagram. The keys in the dictionary are
    the node names, and the values are the node states (0 or 1). The dictionaries are
    sorted by key (node name).

    Parameters
    ----------
    bnet : str
        The Boolean network as a string, with nodes and their update rules
        separated by commas.
    minimal : bool, optional
        If True, only nodes that are minimal trapspaces are included in the output.
        
    Returns
    -------
    tuple[list[str], list[dict[str, int]]]
        A tuple containing a list of node names and a list of dictionaries, where
        each dictionary corresponds to the state of a node in the succession diagram
        of the network.

    Examples
    --------
    >>> get_sd_nodes("A, A | B & C \n B, B & !C \n C, B & !C | !C & !D | !B & C & D \n D, !A & !B & !C & !D | !A & C & D")
    (['A', 'B', 'C', 'D'], [{}, {'B': 0}, {'A': 0, 'B': 0}, {'A': 0, 'B': 0, 'C': 1, 'D': 1}, {'A': 1, 'D': 0}, {'A': 1, 'B': 0, 'D': 0}])
    """
    sd = biobalm.SuccessionDiagram.from_rules(bnet)
    sd.build()

    nodes = sorted(
        [sd.network.get_variable_name(v) for v in sd.network.variables()]
    )

    sd_nodes = []

    for node in sd.node_ids():
        if minimal and not sd.node_is_minimal(node):
            continue
        sd_node = {k: v for k, v in sorted(sd.node_data(node)["space"].items())}
        sd_nodes.append(sd_node)

    sd_nodes = sort_sd_nodes(nodes, sd_nodes)

    return nodes, sd_nodes

def sort_sd_nodes(
    nodes: list[str],
    sd_nodes: list[dict[str, int]],
    DEBUG: bool = False
) -> list[dict[str, int]]:
    """
    Sorts a list of succession diagram nodes based on a custom key derived from the node values.

    This function takes a list of node names and a list of dictionaries representing the 
    states of nodes in a succession diagram. It sorts the list of dictionaries using a custom 
    sorting key such that nodes with absent keys are ranked lowest, followed by nodes with 
    values of 0, then nodes with values of 1. If DEBUG is set to True, the function will 
    perform an additional check to ensure all keys in the dictionaries are valid node names.

    Parameters
    ----------
    nodes : list[str]
        A list of node names that define the order of sorting.
    sd_nodes : list[dict[str, int]]
        A list of dictionaries where each dictionary represents the state of a node in the 
        succession diagram. The keys are node names, and the values are the node states (0 or 1).
    DEBUG : bool, optional
        If set to True, checks if all keys in the dictionaries are in the list of nodes.

    Returns
    -------
    list[dict[str, int]]
        A list of dictionaries representing the sorted succession diagram nodes.

    Examples
    --------
    >>> nodes = ['A', 'B', 'C', 'D']
    >>> sd_nodes = [{'A': 1, 'D': 0}, {'B': 0}, {'A': 1, 'B': 0, 'D': 0}, {'A': 0, 'B': 0}]
    >>> sort_sd_nodes(nodes, sd_nodes)
    [{'B': 0}, {'A': 0, 'B': 0}, {'A': 1, 'D': 0}, {'A': 1, 'B': 0, 'D': 0}]
    """
    if DEBUG:
        # if DEBUG is True, check if all keys in sd_nodes are in nodes
        for sd_node in sd_nodes:
            for key in sd_node.keys():
                if key not in nodes:
                    raise ValueError(f"Key {key} is not in nodes")

    # Custom sorting key function
    def custom_sort_key(d, keys):
        result = []
        for key in keys:
            if key not in d:  # If the key is absent
                result.append(0)  # 0 for absent keys
            else:
                value = d[key]
                if value == 0:
                    result.append(1)  # 1 for value 0
                elif value == 1:
                    result.append(2)  # 2 for value 1
                else:
                    raise ValueError(f"Invalid value {value} for key {key}")
        return tuple(result)

    # Sort the data using the custom sort key
    sorted_sd_nodes = sorted(sd_nodes, key=lambda d: custom_sort_key(d, nodes))

    return sorted_sd_nodes

def generate_states(
    nodes: list[str],
    node_values: dict[str, int],
    valid_exclude_values: list[dict[str, int]],
    DEBUG: bool = False
) -> list[str]:
    """
    Generate all valid binary states for a list of nodes based on specified constraints.

    This function generates all possible binary combinations for nodes not specified
    in the node_values dictionary and filters them based on the valid_exclude_values.
    A state is considered valid if it disagrees with all of the exclude values.
    A state is considered to disagree with an exclude value if any node state disagrees
    with the exclude value.

    Parameters
    ----------
    nodes : list[str]
        A list of node names.
    node_values : dict[str, int]
        A dictionary where keys are node names and values are the fixed binary
        values (0 or 1) for those nodes.
    valid_exclude_values : list[dict[str, int]] or None, optional
        A list of dictionaries, each representing a set of node states to exclude.
        If None, no states are excluded.

    Returns
    -------
    list[str]
        A list of binary strings, each representing a valid state of the nodes.

    Examples
    --------
    >>> nodes = ['A', 'B', 'C', 'D']
    >>> node_values = {'A': 1}
    >>> valid_exclude_values = [{'A': 1, 'B': 1, 'C': 1}, {'D': 0}]
    >>> generate_states(nodes, node_values, valid_exclude_values)
    ['1001', '1011', '1101']
    """

    if DEBUG:
        for values in [node_values, *valid_exclude_values]:
            # check if all keys of node_values and valid_exclude_values are in nodes
            for key in values.keys():
                if key not in nodes:
                    raise ValueError(f"Key {key} is not in nodes")
            # check if all values of node_values and valid_exclude_values are 0 or 1
            for value in values.values():
                if value not in [0, 1]:
                    raise ValueError(f"Value {value} is not 0 or 1")

    # List of nodes to generate binary values for (nodes that aren't in node_values)
    nodes_to_generate = [node for node in nodes if node not in node_values]

    # Get all binary combinations (0 or 1) for the remaining nodes
    binary_combinations = product('01', repeat=len(nodes_to_generate))

    # Generate the binary states
    binary_states = []
    for combination in binary_combinations:
        # Create a copy of node_values and update it with the combination
        state = node_values.copy()

        # Map the combination to the nodes we need to generate values for
        for i, node in enumerate(nodes_to_generate):
            state[node] = int(combination[i])

        # Check if the current state agrees with any of the valid exclude values
        valid_state = True
        for exclude_values in valid_exclude_values:
            valid_state = False
            for node, value in exclude_values.items():
                # if any of state of the node disagrees with exclude_values,
                # it is a valid state, unless it agrees with other exclude_values
                if state[node] != value:
                    valid_state = True
                    break
            if not valid_state:
                break

        if valid_state:
            # Convert the state dictionary to a binary string
            binary_state = ''.join(str(state[node]) for node in nodes)
            binary_states.append(binary_state)

    if DEBUG:
        if len(binary_states) == 0:
            print("No valid states found")

    return binary_states


def get_binary_states(
    nodes: list[str],
    node_values: dict[str, int],
    exclude_values_list: list[dict[str, int]],
    DEBUG: bool = False
) -> list[str]:
    """
    Generate all binary states that agree with node_values and disagree with some of the exclude_values_list.
    Note that only exclude_values that is a subset (hence have more node states specified) of node_values are considered,
    and the rest are ignored.
    For example, if node_values = {'A': 1}, and exclude_values_list = [{'A':1, 'B': 1}, {'A':0, 'B': 0}],
    then only {'A': 1, 'B': 1} is considered, and {'A': 0, 'B': 0} is ignored.

    Parameters
    ----------
    nodes : list[str]
        A list of node names that define the order of sorting.
    node_values : dict[str, int]
        A dictionary where the keys are node names and the values are the node states (0 or 1).
    exclude_values_list : list[dict[str, int]]
        A list of dictionaries where each dictionary represents a set of node states to exclude from the result.
    DEBUG : bool, optional
        If True, perform addtional checks to ensure the input is valid.

    Returns
    -------
    list[str]
        A list of binary strings, each representing a valid state of the Boolean network.

    Examples
    --------
    >>> nodes = ['A', 'B', 'C', 'D']
    >>> node_values = {'A': 1}
    >>> exclude_values_list = [{'A':1, 'B': 1, 'C':1}, {'A':1, 'C': 0}, {'A':0}, {'D': 0}]
    >>> get_binary_states(nodes, node_values, exclude_values_list)
    ['1010', '1011']
    """

    if DEBUG:
        for values in [node_values, *exclude_values_list]:
            # check if all keys of node_values and valid_exclude_values are in nodes
            for key in values.keys():
                if key not in nodes:
                    raise ValueError(f"Key {key} is not in nodes")
        
            # check if all values of node_values and valid_exclude_values are 0 or 1
            for value in values.values():
                if value not in [0, 1]:
                    raise ValueError(f"Value {value} is not 0 or 1")

    # Check which dictionaries in exclude_values_list agree with node_values
    valid_exclude_values: list[dict[str, int]] = []
    
    for exclude_values in exclude_values_list:
        valid = True
        for node, value in node_values.items():
            if node not in exclude_values or value != exclude_values[node]:
                valid = False
                break

        if valid:
            valid_exclude_values.append(exclude_values)

    # Generate all binary states that agree with node_values
    return generate_states(nodes, node_values, valid_exclude_values, DEBUG=DEBUG)

def get_SD_node_states(
    nodes: list[str],
    SD_nodes: list[dict[str, int]],
    DEBUG: bool = False
) -> list[list[str]]:
    """
    Retrieve binary states corresponding to each succession diagram (SD) node.

    This function iterates over a list of SD nodes and generates binary states 
    for each SD node based on the list of provided node names.

    Parameters
    ----------
    nodes: list[str]
        A list of node names.
    SD_nodes: list[dict[str, int]]
        A list of dictionaries where each dictionary represents the state of a node 
        in the succession diagram. The keys are node names, and the values are the node states (0 or 1).
    DEBUG: bool, optional
        If set to True, performs additional checks on the input data.

    Returns
    -------
    list[list[str]]
        A list of lists, where each sublist contains binary state strings corresponding 
        to each SD node.

    Examples
    --------
    >>> nodes = ['A', 'B', 'C', 'D']
    >>> SD_nodes = [{}, {'B': 0}, {'A': 0, 'B': 0}, {'A': 0, 'B': 0, 'C': 1, 'D': 1}, {'A': 1, 'D': 0}, {'A': 1, 'B': 0, 'D': 0}]
    >>> get_SD_node_states(nodes, SD_nodes)
    [['0100', '0101', '0110', '0111', '1101', '1111'], ['1001', '1011'], ['0000', '0001', '0010'], ['0011'], ['1100', '1110'], ['1000', '1010']]
    """
    
    if DEBUG:
        seen = set()
        for SD_node in SD_nodes:
            # check if all keys of SD_node are in nodes
            for key in SD_node.keys():
                if key not in nodes:
                    raise ValueError(f"Key {key} is not in nodes")
            
            # check if all values of SD_node are 0 or 1
            for value in SD_node.values():
                if value not in [0, 1]:
                    raise ValueError(f"Value {value} is not 0 or 1")
        
            # check if SD_node is unique
            dict_frozen_set = frozenset(SD_node.items())
            if dict_frozen_set in seen:
                raise ValueError(f"SD_node {SD_node} is not unique")
            seen.add(dict_frozen_set)

        # check if {} is in SD_nodes
        if {} not in SD_nodes:
            raise ValueError("root node is not in SD_nodes")

    SD_node_states = []  # Initialize a list to store states for each SD node
    
    for SD_node in SD_nodes:
        # Create a list of other SD nodes excluding the current SD node
        other_SD_nodes = [node for node in SD_nodes if node != SD_node]
        
        # Generate binary states for the current SD node
        states = get_binary_states(nodes, SD_node, other_SD_nodes)

        # Append the generated states to the SD_node_states list
        SD_node_states.append(states)
    
    return SD_node_states  # Return the list of states for each SD node


def states_to_indexes(state_groups: list[list[str]], DEBUG: bool = False) -> list[list[int]]:
    """
    Convert a list of binary strings to a list of sorted integers.

    Parameters
    ----------
    state_groups: list[list[str]]
        A list of lists, where each sublist contains binary strings.
    DEBUG: bool, optional
        If True, checks if all state strings have the same length and are mutually exclusive.

    Returns
    -------
    list[list[int]]
        A list of sorted integers, where each sublist corresponds to the binary strings in the input.
    
    Examples
    --------
    >>> states_to_indexes([['0100', '0101', '0110', '0111', '1101', '1111'], ['1001', '1011'], ['0000', '0001', '0010'], ['0011'], ['1100', '1110'], ['1000', '1010']])
    [[4, 5, 6, 7, 13, 15], [9, 11], [0, 1, 2], [3], [12, 14], [8, 10]]
    """

    if DEBUG:
        # Check if all state strings are of the same length
        lengths = [len(state) for state_group in state_groups for state in state_group]
        if len(set(lengths)) != 1:
            raise ValueError("Not all states have the same length")
        
        # Check if all gropus are mutually exclusive
        all_states = [state for state_group in state_groups for state in state_group]
        if len(all_states) != len(set(all_states)):
            raise ValueError("States are not mutually exclusive (duplicates found)")

    # Convert binary strings to integers
    converted = [[int(state, 2) for state in state_group] for state_group in state_groups]
    
    # Sort each sublist
    sorted_states = [sorted(sublist) for sublist in converted]
    
    return sorted_states
