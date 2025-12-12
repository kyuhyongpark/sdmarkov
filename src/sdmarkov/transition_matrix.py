import numpy as np
import networkx as nx

from sdmarkov.helper import check_stg


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


def get_uniform_matrix(n: int, m: int|None = None) -> np.ndarray:
    """
    Generate a uniform matrix of size n x m.

    Parameters
    ----------
    n : int
        The number of rows in the uniform matrix.
    m : int, optional
        The number of columns in the uniform matrix. If not specified, defaults to n.

    Returns
    -------
    np.ndarray
        An n x m matrix where all elements are equal to 1/m.

    Notes
    -----
    A uniform matrix is a matrix where all elements have the same value, in this case 1/n.
    """

    if m is None:
        m = n

    uniform_matrix: np.ndarray = np.ones((n, m))
    return uniform_matrix / m
