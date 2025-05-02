from itertools import product

import biobalm


def get_sd_nodes_and_edges(
    bnet: str, minimal: bool = False, DEBUG: bool = False
) -> tuple[list[str], list[dict[str, int]], list[list[dict[str, int]]]]:
    """
    Given a Boolean network as a string in bnet format, returns a tuple containing
    a list of node names, a list of dictionaries, where each dictionary corresponds
    to the state of a node or an edge in the succession diagram, and a list of lists
    of dictionaries, where each sublist corresponds to the outgoing edges of a node
    in the succession diagram. The keys in the dictionary are the node names, and the
    values are the node states (0 or 1). The dictionaries are sorted by key (node name).

    Parameters
    ----------
    bnet : str
        The Boolean network as a string, with nodes and their update rules
        separated by commas.
    minimal : bool, optional
        If True, only nodes that are minimal trapspaces are included in the output.
    DEBUG : bool, optional
        If True, performs additional checks.
        
    Returns
    -------
    tuple[list[str], list[dict[str, int]], list[list[dict[str, int]]]]
        A tuple containing a list of node names, a list of dictionaries, where
        each dictionary corresponds to the state of a node or an edge in the
        succession diagram, and a list of lists of dictionaries, where each sublist
        corresponds to the outgoing edges of a node in the succession diagram.

   """
    sd = biobalm.SuccessionDiagram.from_rules(bnet)
    sd.expand_bfs()

    nodes = sorted(
        [sd.network.get_variable_name(v) for v in sd.network.variables()]
    )

    sd_nodes = []
    sd_edges = []
    all_groups = []

    for node in sd.node_ids():
        if minimal and not sd.node_is_minimal(node):
            continue
        sd_node = {k: v for k, v in sorted(sd.node_data(node)["space"].items())}
        
        # sd_nodes are intrinsically unique
        sd_nodes.append(sd_node)
        all_groups.append(sd_node)

    for node in sd.node_ids():
        if minimal:
            break

        # get the outgoing edges of the sd node
        for child in sd.node_successors(node, compute=True):
            edge_motifs = sd.edge_all_stable_motifs(node, child, reduced=False)
            sd_edge = []
            for motif in edge_motifs:
                sd_motif = {k: v for k, v in sorted(motif.items())}

                # avoid adding duplicates
                if sd_motif not in all_groups:
                    sd_edge.append(sd_motif)
                    all_groups.append(sd_motif)

            # only add non-empty edge
            if sd_edge:
                sd_edges.append(sort_sd_nodes(nodes, sd_edge, DEBUG=DEBUG))

    # sort the sd nodes
    sd_nodes = sort_sd_nodes(nodes, sd_nodes, DEBUG=DEBUG)

    # sort the sd edges
    first_edges = [sd_edge[0] for sd_edge in sd_edges]
    sorted_edges = sort_sd_nodes(nodes, first_edges, DEBUG=DEBUG)
    sd_edges = sorted(sd_edges, key=lambda x: sorted_edges.index(x[0]))

    return nodes, sd_nodes, sd_edges


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

        # Check if there are any duplicates in the sd nodes
        if len(sd_nodes) != len({frozenset(d.items()) for d in sd_nodes}):
            raise ValueError("There are duplicates in the sd nodes.")

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

    # if DEBUG:
    #     if len(binary_states) == 0:
    #         print("No valid states found")

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

    # if DEBUG:
    #     print(f"node values: {node_values}")
    #     print(f"valid exclude values: {valid_exclude_values}")

    # Generate all binary states that agree with node_values
    return generate_states(nodes, node_values, valid_exclude_values, DEBUG=DEBUG)


def get_sd_group_states(
    nodes: list[str],
    sd_nodes: list[dict[str, int]],
    sd_edges: list[list[dict[str, int]]],
    extra_groups: list[dict[str, int]] = [],
    DEBUG: bool = False
) -> tuple[bool, list[list[str]], dict[str, list[list[str]]]]:
    """
    Retrieve binary states corresponding to each succession diagram (SD) node.

    This function iterates over a list of SD nodes and generates binary states 
    for each SD node based on the list of provided node names. It also ensures 
    that each SD node and edge has a unique set of binary states.

    Parameters
    ----------
    nodes: list[str]
        A list of node names.
    sd_nodes: list[dict[str, int]]
        A list of dictionaries where each dictionary represents the state of a node 
        in the succession diagram. The keys are node names, and the values are the node states (0 or 1).
    sd_edges: list[list[dict[str, int]]]
        A list of lists where each sublist represents outgoing edges of a node in the 
        succession diagram. Each edge is a dictionary similar to sd_nodes.
    extra_groups: list[dict[str, int]], optional
        A list of dictionaries where each dictionary represents the state.
        The keys are node names, and the values are the node states (0 or 1).
        May be needed to ensure that all states are unique.
    DEBUG: bool, optional
        If set to True, performs additional checks on the input data.

    Returns
    -------
    tuple[bool, list[list[str]], dict[str, list[list[str]]]]
        A tuple containing:
        - A boolean indicating whether the total number of states equals 2^N (N being number of nodes).
        - A list of lists, where each sublist contains binary state strings corresponding 
          to each SD node and edge.
        - A dictionary of duplicate states and their corresponding groups, if any.

    Notes
    -----
    It may happen that certain SD nodes do not have a corresponding binary state.
    In that case, the corresponding sublist in the returned list will be empty.

    """
    
    if DEBUG:
        seen = set()
        for sd_node in sd_nodes:
            # check if all keys of SD_node are in nodes
            for key in sd_node.keys():
                if key not in nodes:
                    raise ValueError(f"Key {key} is not in nodes")
            
            # check if all values of SD_node are 0 or 1
            for value in sd_node.values():
                if value not in [0, 1]:
                    raise ValueError(f"Value {value} is not 0 or 1")
        
            # check if SD_node is unique
            dict_frozen_set = frozenset(sd_node.items())
            if dict_frozen_set in seen:
                raise ValueError(f"SD_node {sd_node} is not unique")
            seen.add(dict_frozen_set)

        # Check if SD_nodes is not empty
        if not sd_nodes:
            raise ValueError("SD_nodes is empty")

    sd_group_states = []  # Initialize a list to store states for each group
    
    all_subspaces = sd_nodes.copy()
    for sd_edge in sd_edges:
        for motif in sd_edge:
            if motif not in all_subspaces:
                all_subspaces.append(motif)
    for extra_group in extra_groups:
        if extra_group not in all_subspaces:
            all_subspaces.append(extra_group)

    for sd_node in sd_nodes:
        # Create a list of other SD nodes excluding the current SD node
        other_subspaces = all_subspaces.copy()
        other_subspaces.remove(sd_node)
        
        # Generate binary states for the current SD node
        states = get_binary_states(nodes, sd_node, other_subspaces, DEBUG=DEBUG)

        # Append the generated states to the SD_node_states list
        sd_group_states.append(states)
    
    for sd_edge in sd_edges:
        edge_states = []
        for motif in sd_edge:
            # Do not add motif if it is already in SD_nodes
            if motif in sd_nodes:
                continue

            other_subspaces = all_subspaces.copy()
            other_subspaces.remove(motif)
            states = get_binary_states(nodes, motif, other_subspaces, DEBUG=DEBUG)
            for state in states:
                if state not in edge_states:
                    edge_states.append(state)

        sd_group_states.append(sorted(edge_states))
    
    for extra_group in extra_groups:
        other_subspaces = all_subspaces.copy()
        other_subspaces.remove(extra_group)
        states = get_binary_states(nodes, extra_group, other_subspaces, DEBUG=DEBUG)
        sd_group_states.append(states)

    # The number of all states must be equal to 2**N, where N is the number of nodes
    N = len(nodes)
    total_states = sum(len(states) for states in sd_group_states)
    if total_states != 2**N:

        # find the duplicate states
        all_states = [state for states in sd_group_states for state in states]

        duplicate_states = []
        for state in all_states:
            if all_states.count(state) > 1 and state not in duplicate_states:
                duplicate_states.append(state)

        if duplicate_states:
            # if DEBUG:
            #     print(f"{sd_nodes=}")
            #     print(f"{sd_edges=}")

            duplicate_dict = {}

            for state in duplicate_states:
                duplicate_dict[state] = []
                # if DEBUG:
                #     print(f"Duplicate state: {state}")
                for i, state_group in enumerate(sd_group_states):
                    if state in state_group:

                        if i in range(len(sd_nodes)):
                            duplicate_dict[state].append(sd_nodes[i])
                            # if DEBUG:
                            #     print(f"SD node: {sd_nodes[i]}")
                        elif i in range(len(sd_nodes), len(sd_nodes) + len(sd_edges)):
                            duplicate_dict[state].append(sd_edges[i - len(sd_nodes)])
                            # if DEBUG:
                            #     print(f"SD edge: {sd_edges[i - len(sd_nodes)]}")
                        else:
                            duplicate_dict[state].append(extra_groups[i - len(sd_nodes) - len(sd_edges)])
                            # if DEBUG:
                            #     print(f"Extra group: {extra_groups[i - len(sd_nodes) - len(sd_edges)]}")

            return False, sd_group_states, duplicate_dict
    
    return True, sd_group_states, {}
