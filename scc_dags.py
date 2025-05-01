import networkx as nx

def get_scc_dag(stg: nx.DiGraph) -> nx.DiGraph:
    """
    Convert a state transition graph (STG) to a strongly connected component (SCC) directed acyclic graph (DAG).

    The SCC DAG is constructed by representing each strongly connected component in the STG as a unique node in the DAG.
    The edges between the nodes in the DAG correspond to edges between the SCCs in the STG.

    Parameters
    ----------
    stg : networkx DiGraph
        The state transition graph.

    Returns
    -------
    dag : networkx DiGraph
        The SCC DAG.
    """
    # Step 1: Find the strongly connected components (SCCs)
    sccs = list(nx.strongly_connected_components(stg))

    # Step 2: Create a new DAG
    dag = nx.DiGraph()

    # Create a mapping from states to SCCs
    state_to_scc = {}
    for idx, scc in enumerate(sccs):
        # Each SCC will be represented by a unique node in the DAG
        dag.add_node(idx, states=scc)
        for state in scc:
            state_to_scc[state] = idx  # Map each state to its SCC ID

    # Step 3: Add edges to the DAG
    # We go through each edge in the original graph and if the source and destination belong to different SCCs,
    # we add an edge between the corresponding SCCs in the DAG.
    for u, v in stg.edges():
        scc_u = state_to_scc[u]
        scc_v = state_to_scc[v]
        if scc_u != scc_v:
            dag.add_edge(scc_u, scc_v)
    
    return dag


def get_ordered_states(scc_dag: nx.DiGraph, as_indices: bool = False, DEBUG: bool = False) -> list[list[int]]:
    """
    Returns the ordered list of states in the SCC DAG.
    The states are ordered according to the topological order of the SCC DAG.

    Attractor states are moved to the end of the list, and are sorted by their first state.

    Parameters
    ----------
    scc_dag : networkx DiGraph
        The SCC DAG.
    as_indices : bool, optional
        If True, the states are returned as a list of decimal indices,
        otherwise as a list of binary strings.
    DEBUG : bool, optional
        If True, performs additional checks.

    Returns
    -------
    ordered_states : list
        The ordered list of states in the SCC DAG.
    """
    # Get the topological order of the SCC DAG
    topological_order = list(nx.topological_sort(scc_dag))

    # Get the attractors
    attractors = [node for node, out_degree in scc_dag.out_degree() if out_degree == 0]

    # Remove the attractors from the topological order
    non_attractors = [node for node in topological_order if node not in attractors]

    # Build the final ordered list of states
    ordered_states = []
    for scc_id in non_attractors:
        # Get the states in the SCC and sort them
        scc_states = sorted(scc_dag.nodes[scc_id]['states'])
        ordered_states.append(scc_states)

    attractor_states = []
    for scc_id in attractors:
        # Get the states in the SCC and sort them
        scc_states = sorted(scc_dag.nodes[scc_id]['states'])
        attractor_states.append(scc_states)
    attractor_states.sort(key=lambda x: x[0])

    ordered_states.extend(attractor_states)

    if as_indices:
        # Convert list of binary string to list of decimals
        ordered_indices = []
        for binary_strings in ordered_states:
            index_list = []
            for binary_string in binary_strings:
                decimal_value = int(binary_string, 2)
                index_list.append(decimal_value)
            ordered_indices.append(index_list)
        return ordered_indices

    return ordered_states


def get_attractor_states(scc_dag: nx.DiGraph, as_indices: bool = False, DEBUG: bool = False) -> list[list[int]]:
    """
    Retrieve the attractor states from a given SCC DAG.

    Attractor states correspond to the sink nodes in the SCC DAG, which have no outgoing edges.

    Parameters
    ----------
    scc_dag : networkx DiGraph
        The SCC DAG to analyze.
    as_indices : bool, optional
        If True, the attractor states are returned as decimal indices,
        otherwise as binary strings.

    Returns
    -------
    attractor_states : list
        A list of attractor states in the SCC DAG.
        Each attractor is represented by the states within it, either as binary strings or decimal indices.
    """

    # Identify attractors (sink nodes in the DAG)
    attractors = [node for node, out_degree in scc_dag.out_degree() if out_degree == 0]

    attractor_states = []
    for node in attractors:
        attractor_states.append(sorted(scc_dag.nodes[node]['states']))

    if DEBUG:
        if not attractor_states:
            raise ValueError("No attractors found in the SCC DAG.")

    # sort the attractor states by the first state
    attractor_states.sort(key=lambda x: x[0])

    if as_indices:
        # Convert list of binary strings to list of decimals
        attractor_indices = []
        for binary_strings in attractor_states:
            index_list = []
            for binary_string in binary_strings:
                decimal_value = int(binary_string, 2)
                index_list.append(decimal_value)
            attractor_indices.append(index_list)
        return attractor_indices

    return attractor_states
