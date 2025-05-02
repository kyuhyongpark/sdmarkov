import numpy as np
import networkx as nx

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

def check_transition_matrix(transition_matrix: np.ndarray, compressed: bool=False, partial: bool=False) -> None:
    """
    Validate the structure and properties of a transition matrix.

    Parameters
    ----------
    transition_matrix : np.ndarray, shape (2^N, 2^N)
        The matrix representing state transitions.
    compressed : bool, optional
        If True, the matrix is not required to have dimensions of 2^N.

    Raises
    ------
    ValueError
        If the matrix does not meet the specified criteria.

    Notes
    -----
    - The matrix should be square.
    - All elements must be between 0 and 1.
    - Each row must sum to 1.
    - If `compressed` is False, the number of rows/columns should be 2^N.
    """

    # Check that the elements of the array are between 0 and 1, with some tolerance
    if not np.all(transition_matrix >= 0 - 1e-16) or not np.all(transition_matrix <= 1 + 1e-16):
        raise ValueError("All elements of the matrix must be between 0 and 1. Max: {}, Min: {}".format(np.max(transition_matrix), np.min(transition_matrix)))

    if not partial:
        # Check that the matrix is square
        if transition_matrix.shape[0] != transition_matrix.shape[1]:
            raise ValueError("The matrix must be square.")

        # Check that every row of the matrix sums to 1
        if not np.allclose(np.sum(transition_matrix, axis=1), np.ones(transition_matrix.shape[1])):
            raise ValueError("Every row of the matrix must sum to 1.")

    if not compressed and not partial:
        # Check if the length of the matrix is 2^N
        N = int(np.log2(transition_matrix.shape[0]))
        if 2**N != transition_matrix.shape[0]:
            raise ValueError("The length of the matrix must be 2^N.")


def get_transition_matrix(stg: nx.DiGraph, update: str = "asynchronous", DEBUG: bool = False) -> np.ndarray:
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

    # Initialize the transition matrix, with datatype float64
    transition_matrix = np.zeros((2**N,2**N), dtype=np.float64)

    states = sorted(list(stg.nodes()))

    for state in states:
        index = states.index(state)

        out_edges = stg.out_edges(state)

        for out_edge in out_edges:
            out_index = states.index(out_edge[1])

            if out_index == index:
                continue
            
            if update == "asynchronous":
                transition_matrix[index][out_index] = 1/N
            elif update == "synchronous":
                transition_matrix[index][out_index] = 1

        transition_matrix[index][index] = 1 - np.sum(transition_matrix[index])

    # Ensure the transition matrix is a valid probability matrix
    transition_matrix[transition_matrix < 0] = 0

    transition_matrix = transition_matrix / np.sum(transition_matrix, axis=1, keepdims=True)

    if DEBUG:
        # Check if the transition matrix is a valid probability matrix
        if not np.allclose(np.sum(transition_matrix, axis=1), np.ones(2**N)):
            raise ValueError("The transition matrix is not a valid probability matrix.")

    return transition_matrix

def get_hamming_distance_matrix(N: int|None = None, stg: nx.DiGraph|None = None, DEBUG: bool = False) -> np.ndarray:
    """
    Construct a Hamming distance matrix from a state transition graph.

    Parameters
    ----------
    N : int, optional
        The number of nodes in the Boolean network. If None, it is inferred from stg.
    stg : networkx DiGraph, optional
        The state transition graph. If None, N must be specified.

    Returns
    -------
    hamming_distance_matrix : numpy array, shape (2^N, 2^N)
        The Hamming distance matrix. The entry at row i and column j is the Hamming distance between state i and state j.

    Notes
    -----
    The Hamming distance matrix is a square matrix of size 2^N, where N is the number of nodes in the Boolean network.
    The Hamming distance between two states is the number of bits that are different between the two states.
    """

    if N == None:
        if stg == None:
            raise ValueError("Either N or stg must be specified.")
        # If the state transition graph is empty, return None
        if stg.number_of_nodes() == 0:
            if DEBUG:
                print("The state transition graph is empty.")
            return None

        # Perform basic checks if DEBUGGING
        if DEBUG:
            check_stg(stg)

        # Get the number of nodes in the Boolean network
        N = len(list(stg.nodes())[0])

    array_00 = np.zeros((1,1))
    array_01 = np.ones((1,1))

    for i in range(N):
        array_10 = array_01
        array_11 = array_00

        top_half = np.hstack((array_00, array_01))
        bottom_half = np.hstack((array_10, array_11))
                
        # Step 4: Create the result by stacking top and bottom vertically (2n x 2n)
        array_00 = np.vstack((top_half, bottom_half))
        array_01 = array_00 + 1

    hamming_distance_matrix = array_00

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

    markov_chain = nx.DiGraph()

    group_names = [str(i) for i, group in enumerate(group_indices) if group]

    if DEBUG:
        if len(group_names) != compressed_transition_matrix.shape[0]:
            raise ValueError("Number of group names does not match number of rows in compressed transition matrix")

    for i in range(compressed_transition_matrix.shape[0]):
        for j in range(compressed_transition_matrix.shape[1]):
            if compressed_transition_matrix[i][j] > 0:
                markov_chain.add_edge(group_names[i], group_names[j], weight=compressed_transition_matrix[i][j])

    return markov_chain
