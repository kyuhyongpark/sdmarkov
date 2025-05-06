import networkx as nx
import numpy as np

from helper import indices_to_states


def get_stg(transition_matrix: np.ndarray, DEBUG: bool = False) -> nx.DiGraph:
    """
    Construct a state transition graph from a transition matrix.

    Parameters
    ----------
    transition_matrix : numpy array, shape (2^N, 2^N)
        The transition matrix. The entry at row i and column j is the probability of transitioning from state i to state j.

    DEBUG : bool, optional
        If set to True, performs additional checks on the input data.

    Returns
    -------
    stg : networkx DiGraph
        The state transition graph.

    Notes
    -----
    The state transition graph is a directed graph where each node represents a state and each edge represents a transition between two states.
    For consistency, we do not allow self-loops in the state transition graph, unless the state is a terminal state. This choice is purely for convenience.
    """

    # If the transition matrix is empty, return an empty graph
    if transition_matrix.size == 0:
        if DEBUG:
            print("Transition matrix is empty")
        return nx.DiGraph()

    # Perform basic checks if DEBUGGING
    if DEBUG:
        # Transition matrix should be a square matrix
        if transition_matrix.shape[0] != transition_matrix.shape[1]:
            raise ValueError("Transition matrix should be a square matrix")

        # all elements of transition matrix should be between 0 and 1
        for i in range(transition_matrix.shape[0]):
            for j in range(transition_matrix.shape[1]):
                if transition_matrix[i][j] < 0 or transition_matrix[i][j] > 1:
                    raise ValueError("all elements of transition matrix should be between 0 and 1")

        # all rows of the transition matrix should sum to 1
        for i in range(transition_matrix.shape[0]):
            if not np.isclose(np.sum(transition_matrix[i]), 1):
                raise ValueError("all rows of the transition matrix should sum to 1")

    stg = nx.DiGraph()

    for i in range(transition_matrix.shape[0]):
        for j in range(transition_matrix.shape[1]):

            binary_i = bin(i)[2:]  # Convert to binary and remove '0b' prefix
            binary_i = binary_i.zfill(int(np.log2(transition_matrix.shape[0])))  # Add leading zeros if necessary

            binary_j = bin(j)[2:]  # Convert to binary and remove '0b' prefix
            binary_j = binary_j.zfill(int(np.log2(transition_matrix.shape[0])))  # Add leading zeros if necessary            

            stg.add_node(binary_i)
            stg.add_node(binary_j)

            if transition_matrix[i][j] == 1 and i == j:
                stg.add_edge(binary_i, binary_j, weight=transition_matrix[i][j])
            elif transition_matrix[i][j] > 0 and i != j:
                stg.add_edge(binary_i, binary_j, weight=transition_matrix[i][j])

    return stg


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


def get_markov_chain(compressed_transition_matrix: np.ndarray, group_indices: list, DEBUG: bool = False) -> nx.DiGraph:
    """
    Construct a Markov chain from a compressed transition matrix.

    Parameters
    ----------
    compressed_transition_matrix : np.ndarray
        The compressed transition matrix. The entry at row i and column j is the probability of transitioning from group i to group j.

    group_indices : list
        A list of lists containing the indices corresponding to each group in the compressed matrix.

    DEBUG : bool, optional
        If set to True, performs additional checks on the input data.

    Returns
    -------
    markov_chain : networkx.DiGraph
        The Markov chain represented as a directed graph.

    Notes
    -----
    The function assumes that the compressed transition matrix is already validated for its dimensions and properties.
    """

    if compressed_transition_matrix.size == 0:
        if DEBUG:
            print("Compressed transition matrix is empty")
        return nx.DiGraph()

    if DEBUG:
        if compressed_transition_matrix.shape[0] != compressed_transition_matrix.shape[1]:
            raise ValueError("Compressed transition matrix should be a square matrix")

        if not np.all((compressed_transition_matrix >= 0) & (compressed_transition_matrix <= 1)):
            raise ValueError("All elements of the compressed transition matrix should be between 0 and 1")

        if not np.allclose(np.sum(compressed_transition_matrix, axis=1), 1):
            raise ValueError("All rows of the compressed transition matrix should sum to 1")

    # Get the number of nodes
    all_indices = []
    for group in group_indices:
        all_indices += group

    if DEBUG:
        if len(all_indices) == 0:
            raise ValueError("Group indices are empty")

        if len(all_indices) != 2 ** int(np.log2(len(all_indices))):
            raise ValueError("Number of indices is not a power of 2")
        
    N = int(np.log2(len(all_indices)))

    # Get the group states
    group_states = indices_to_states(group_indices, N, DEBUG=DEBUG)

    # Create the Markov chain
    markov_chain = nx.DiGraph()

    # Add nodes to the Markov chain
    group_names = []
    for i in range(len(group_indices)):
        # check if group_indices[i] is not empty
        if group_indices[i]:
            markov_chain.add_node("G" + str(i), indices = group_indices[i], states = group_states[i])
            group_names.append("G" + str(i))

    if DEBUG:
        if len(group_names) != compressed_transition_matrix.shape[0]:
            raise ValueError("Number of group names does not match number of rows in compressed transition matrix")

    # Add edges to the Markov chain
    # (j, k) is an edge from the j-th non-empty group to the k-th non-empty group
    for j in range(compressed_transition_matrix.shape[0]):
        for k in range(compressed_transition_matrix.shape[1]):
            if compressed_transition_matrix[j][k] > 0:
                markov_chain.add_edge(group_names[j], group_names[k], weight=compressed_transition_matrix[j][k])

    return markov_chain