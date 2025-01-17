import numpy as np
import networkx as nx

def check_stg(stg: nx.DiGraph) -> None:

    """
    Check if a given state transition graph is valid.

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

def get_transition_matrix(stg: nx.DiGraph, DEBUG: bool = False) -> np.ndarray:
    """
    Construct a transition matrix from a state transition graph.

    Parameters
    ----------
    stg : networkx DiGraph
        The state transition graph.
    DEBUG : bool, optional
        If set to True, performs additional checks on the input data.

    Returns
    -------
    transition_matrix : numpy array, shape (2^N, 2^N)
        The transition matrix. The entry at row i and column j is the probability of transitioning from state i to state j.

    Notes
    -----
    The transition matrix is a square matrix of size 2^N, where N is the number of nodes in the Boolean network.
    """

    # If the state transition graph is empty, return None
    if stg.number_of_nodes() == 0:
        if DEBUG:
            print("The state transition graph is empty.")
        return None

    N = len(list(stg.nodes())[0])

    # Perform basic checks if DEBUGGING
    if DEBUG:
        check_stg(stg)

    # Initialize the transition matrix
    transition_matrix = np.zeros((2**N,2**N))

    states = sorted(list(stg.nodes()))

    for state in states:
        index = states.index(state)

        out_edges = stg.out_edges(state)

        for out_edge in out_edges:
            out_index = states.index(out_edge[1])

            if out_index == index:
                continue

            transition_matrix[index][out_index] = 1/N

        transition_matrix[index][index] = 1 - np.sum(transition_matrix[index])

    if DEBUG:
        # Check if the transition matrix is a valid probability matrix
        if not np.allclose(np.sum(transition_matrix, axis=1), np.ones(2**N)):
            raise ValueError("The transition matrix is not a valid probability matrix.")

    return transition_matrix

def get_hamming_distance_matrix(stg: nx.DiGraph, DEBUG: bool = False) -> np.ndarray:
    """
    Construct a Hamming distance matrix from a state transition graph.

    Parameters
    ----------
    stg : networkx DiGraph
        The state transition graph.

    Returns
    -------
    hamming_distance_matrix : numpy array, shape (2^N, 2^N)
        The Hamming distance matrix. The entry at row i and column j is the Hamming distance between state i and state j.

    Notes
    -----
    The Hamming distance matrix is a square matrix of size 2^N, where N is the number of nodes in the Boolean network.
    The Hamming distance between two states is the number of bits that are different between the two states.
    """

    # If the state transition graph is empty, return None
    if stg.number_of_nodes() == 0:
        if DEBUG:
            print("The state transition graph is empty.")
        return None

    # Perform basic checks if DEBUGGING
    if DEBUG:
        check_stg(stg)

    # Get the list of states in the state transition graph
    states = sorted(list(stg.nodes()))

    # Get the number of nodes in the Boolean network
    N = len(list(stg.nodes())[0])

    # Initialize the Hamming distance matrix
    hamming_distance_matrix = np.zeros((2**N,2**N))

    # Iterate over all pairs of states
    for state1 in states:
        # Get the index of state1 in the sorted list of states
        index1 = states.index(state1)

        for state2 in states:
            # Get the index of state2 in the sorted list of states
            index2 = states.index(state2)

            # Calculate the Hamming distance between state1 and state2
            for i in range(N):
                if state1[i] != state2[i]:
                    hamming_distance_matrix[index1][index2] += 1

    return hamming_distance_matrix

def get_bitflip_matrix(hd: np.ndarray, size: int, DEBUG: bool = False) -> np.ndarray:
    """
    Construct a bitflip matrix from a Hamming distance matrix.

    Parameters
    ----------
    hd : numpy array, shape (2^N, 2^N)
        The Hamming distance matrix. The entry at row i and column j is the Hamming distance between state i and state j.

    size : int
        The size of the bitflip.

    Returns
    -------
    bitflip_matrix : numpy array, shape (2^N, 2^N)
        The bitflip matrix. The entry at row i and column j is the probability of transitioning from state i to state j by flipping size bits.

    Notes
    -----
    Bitflip happens only for the given size.
    Generate multiple bitflip matrices and combine them when needed.
    The probabilities are distributed uniformly over the bitflips of the same size.
    """

    # If the Hamming distance matrix is empty, return None
    if hd.size == 0:
        return None
    
    # size should be between 1 and N
    if size < 1 or size > np.max(hd):
        raise ValueError("size should be between 1 and N")

    # Perform basic checks if DEBUGGING
    if DEBUG:
        # Hamming distance matrix should be a square matrix
        if hd.shape[0] != hd.shape[1]:
            raise ValueError("Hamming distance matrix should be a square matrix")
        
        # max(hd) should be equal to N
        if 2**np.max(hd) != hd.shape[0]:
            raise ValueError("max(hd) should be equal to N")
        
        # all elements of hd should be integers between 0 and N
        for i in range(hd.shape[0]):
            for j in range(hd.shape[1]):
                if not isinstance(hd[i][j], int) or hd[i][j] < 0 or hd[i][j] > np.max(hd):
                    raise ValueError("all elements of hd should be integers between 0 and N")
        
        # Each row of the Hamming distance matrix should have all values between 0 and N
        for i in range(hd.shape[0]):
            numbers = set(hd[i])
            if len(numbers) != np.max(hd):
                raise ValueError("Each row of the Hamming distance matrix should have all values between 0 and N")

    bitflip_matrix = np.zeros((len(hd), len(hd)))

    for i in range(len(hd)):
        for j in range(len(hd)):
            if hd[i][j] == size:
                bitflip_matrix[i][j] = 1

    for i in range(len(hd)):
        bitflip_matrix[i] = bitflip_matrix[i] / np.sum(bitflip_matrix[i])

    return bitflip_matrix


def get_identity_matrix(n: int) -> np.ndarray:
    """
    Generate a identity matrix of size n x n.

    Parameters
    ----------
    n : int
        The size of the identity matrix, which is a square matrix.

    Returns
    -------
    numpy.ndarray
        An n x n identity matrix with ones on the main diagonal and zeros elsewhere.

    """

    identity_matrix: np.ndarray = np.zeros((n,n))
    for i in range(n):
        identity_matrix[i][i] = 1
    return identity_matrix


def get_uniform_matrix(n: int) -> np.ndarray:
    """
    Generate a uniform matrix of size n x n.

    Parameters
    ----------
    n : int
        The size of the uniform matrix, which is a square matrix.

    Returns
    -------
    np.ndarray
        An n x n matrix where all elements are equal to 1/n.

    Notes
    -----
    A uniform matrix is a matrix where all elements have the same value, in this case 1/n.
    """

    uniform_matrix: np.ndarray = np.ones((n, n))
    return uniform_matrix / n
