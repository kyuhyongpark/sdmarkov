import warnings
from itertools import product

import biobalm


def get_sd_nodes_and_edges(
    bnet: str,
    minimal: bool = False,
    DEBUG: bool = False,
) -> tuple[
    list[str],
    list[dict[str, int]],
    list[dict[str, int]],
]:
    """
    Extract symbolic trap spaces that corresponds to a succession diagram from a Boolean network.

    Parameters
    ----------
    bnet : str
        Boolean network in .bnet format.
    minimal : bool, optional
        If True, return only minimal trap spaces.
    DEBUG : bool, optional
        If True, enable additional internal checks and diagnostics.

    Returns
    -------
    nodes : list[str]
        Canonical node ordering.
    node_trap_spaces : list[dict[str, int]]
        Trap spaces of succession diagram nodes.
    edge_trap_spaces : list[dict[str, int]]
        Trap spaces of succession diagram edges.

    Notes
    -----
    - Each trap space is a partial assignment.
    - `edge_trap_spaces` is flat; provenance is intentionally discarded.
    - Trap spaces may overlap.
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
            for motif in edge_motifs:
                sd_motif = {k: v for k, v in sorted(motif.items())}

                # avoid adding duplicates
                if sd_motif not in all_groups:
                    sd_edges.append(sd_motif)
                    all_groups.append(sd_motif)

    # sort the sd nodes and edges
    sd_nodes = sort_sd_nodes(nodes, sd_nodes, DEBUG=DEBUG)
    sd_edges = sort_sd_nodes(nodes, sd_edges, DEBUG=DEBUG)

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


def trap_spaces_overlap(
    a: dict[str, int],
    b: dict[str, int],
) -> bool:
    """
    Return True if two trap spaces overlap (i.e. are jointly satisfiable).

    Two trap spaces overlap if there is no variable that is assigned
    different values in a and b.
    """
    # Iterate over the smaller dict for efficiency
    if len(a) > len(b):
        a, b = b, a

    for var, val in a.items():
        if var in b and b[var] != val:
            return False

    return True


def intersect_trap_spaces(
    a: dict[str, int],
    b: dict[str, int],
) -> dict[str, int] | None:
    """
    Return the intersection trap space if consistent, else None.

    The intersection is the union of assignments if no conflicts exist.
    """
    intersection = dict(a)

    for var, val in b.items():
        if var in intersection:
            if intersection[var] != val:
                return None
        else:
            intersection[var] = val

    return intersection


def close_trap_spaces_under_intersection(
    trap_spaces: list[dict[str, int]],
    DEBUG: bool = False,
) -> list[dict[str, int]]:
    """
    Close a collection of trap spaces under intersection.
    """
    # Work with a list + set of frozensets for fast membership tests
    closed = list(trap_spaces)
    seen = {frozenset(ts.items()) for ts in closed}

    changed = True
    while changed:
        changed = False
        n = len(closed)

        for i in range(n):
            for j in range(i + 1, n):
                a = closed[i]
                b = closed[j]

                inter = intersect_trap_spaces(a, b)
                if inter is None:
                    continue

                key = frozenset(inter.items())
                if key not in seen:
                    if DEBUG:
                        print(f"Adding intersection: {inter}")
                    closed.append(inter)
                    seen.add(key)
                    changed = True

    return closed


def validate_no_missing_intersections(
    trap_spaces: list[dict[str, int]],
    DEBUG: bool = False,
) -> bool:
    """
    Validate that all overlapping trap spaces have their intersection included.
    """
    # Precompute keys for fast lookup
    trap_space_keys = {frozenset(ts.items()) for ts in trap_spaces}

    valid = True

    n = len(trap_spaces)
    for i in range(n):
        for j in range(i + 1, n):
            a = trap_spaces[i]
            b = trap_spaces[j]

            inter = intersect_trap_spaces(a, b)
            if inter is None:
                continue

            key = frozenset(inter.items())
            if key not in trap_space_keys:
                valid = False
                if DEBUG:
                    print(
                        "Missing intersection:",
                        f"a={a}, b={b}, intersection={inter}",
                    )

    return valid


def build_sd_trap_spaces(
    bnet: str,
    minimal: bool = False,
    DEBUG: bool = False,
) -> tuple[list[str], list[dict[str, int]]]:
    """
    Build symbolic trap spaces derived from stable-decision (SD) motifs.

    The procedure prioritizes edge-derived trap spaces, since most overlaps
    originate from edges. A full closure is only performed if needed.
    """
    # Extract SD-derived trap spaces
    nodes, node_ts, edge_ts = get_sd_nodes_and_edges(bnet, minimal=minimal, DEBUG=DEBUG)

    # Stage 1: close edge-derived trap spaces
    closed_edges = close_trap_spaces_under_intersection(edge_ts, DEBUG=DEBUG)

    # Stage 2: merge node-level trap spaces
    all_ts = closed_edges + [
        ts for ts in node_ts if ts not in closed_edges
    ]

    # Stage 3: validate once
    if not validate_no_missing_intersections(all_ts, DEBUG=DEBUG):
        warnings.warn(
            "WARNING: Missing intersections after merging node trap spaces. "
            "Performing full closure.",
            RuntimeWarning
        )
        all_ts = close_trap_spaces_under_intersection(all_ts, DEBUG=DEBUG)

    all_ts = sort_sd_nodes(nodes, all_ts, DEBUG=DEBUG)

    return nodes, all_ts



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


def assign_states_to_trap_spaces(
    nodes: list[str],
    trap_spaces: list[dict[str, int]],
    DEBUG: bool = False,
) -> list[list[str]]:
    """
    Assign concrete binary states to each trap space.

    Parameters
    ----------
    nodes : list[str]
        Ordered list of node names defining bit positions.
    trap_spaces : list[dict[str, int]]
        Trap spaces as partial assignments, assumed to be
        closed under intersection.
    DEBUG : bool, optional
        If True, assert full coverage and no duplication.

    Returns
    -------
    trap_space_states : list[list[str]]
        For each trap space, the list of binary state strings
        belonging to it, in the same order as `trap_spaces`.
    """

    trap_space_states: list[list[str]] = []

    for ts in trap_spaces:
        # All other trap spaces act as exclusions
        other_spaces = [o for o in trap_spaces if o is not ts]

        states = get_binary_states(
            nodes=nodes,
            node_values=ts,
            exclude_values_list=other_spaces,
            DEBUG=DEBUG,
        )

        trap_space_states.append(states)

    if DEBUG:
        N = len(nodes)

        all_states = [s for group in trap_space_states for s in group]

        # Coverage check
        if len(all_states) != 2**N:
            raise AssertionError(
                f"Total number of states {len(all_states)} "
                f"does not equal 2^{N}"
            )

        # Uniqueness check
        if len(set(all_states)) != len(all_states):
            raise AssertionError("Duplicate states detected across trap spaces")

    return trap_space_states

