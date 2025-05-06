import networkx as nx
import numpy as np

from helper import indices_to_states


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