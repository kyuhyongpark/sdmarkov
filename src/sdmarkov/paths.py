import random

import networkx as nx
import numpy as np

from sdmarkov.transition_matrix import get_transition_matrix


def get_all_paths(markov_chain: nx.DiGraph, DEBUG: bool = False) -> list[tuple]:
    """
    Retrieve all unique simple edge paths in a Markov chain.

    This function computes all simple edge paths between distinct pairs of nodes
    in the given Markov chain. It ensures that the paths are unique, sorted
    first by their length and then lexicographically.

    Parameters
    ----------
    markov_chain : networkx DiGraph
        The Markov chain represented as a directed graph.
    DEBUG : bool, optional
        If set to True, performs additional checks on the input data.

    Returns
    -------
    all_paths : list of tuples
        A list of tuples, each representing a unique simple path in the
        Markov chain. Each path is represented as a tuple of nodes.
    """

    all_paths = set()
    for source in markov_chain.nodes:
        for target in markov_chain.nodes:
            if source == target:
                continue

            for path in nx.all_simple_edge_paths(markov_chain, source, target):
                # Start with the first tuple
                path_list = [path[0][0], path[0][1]]

                # Iterate over the remaining tuples and add the second element of each
                for i in range(1, len(path)):
                    # Check that the second element of the previous tuple matches the first element of the next
                    if path[i-1][1] == path[i][0]:
                        path_list.append(path[i][1])
                    else:
                        raise ValueError("The second element of one tuple does not match the first element of the next tuple.")

                all_paths.add(tuple(path_list))

    all_paths = sorted(list(all_paths), key=lambda x: (len(x), *x))

    return all_paths


def get_all_shortest_paths(markov_chain: nx.DiGraph, cutoff: int = 0, to_attractors: bool = False, DEBUG: bool = False) -> list[tuple]:
    """
    Retrieve all unique simple edge paths in a Markov chain.

    This function computes all simple edge paths between distinct pairs of nodes
    in the given Markov chain. It ensures that the paths are unique, sorted
    first by their length and then lexicographically.

    Parameters
    ----------
    markov_chain : networkx DiGraph
        The Markov chain represented as a directed graph.
    cutoff : int, optional
        The maximum length of the paths to consider.
    DEBUG : bool, optional
        If set to True, performs additional checks on the input data.

    Returns
    -------
    all_paths : list of tuples
        A list of tuples, each representing a unique simple path in the
        Markov chain. Each path is represented as a tuple of nodes.
    """

    shortest_paths = set()
    for source in markov_chain.nodes:
        for target in markov_chain.nodes:
            if source == target:
                continue
            
            try:
                current_cutoff = cutoff+1
                for path in nx.shortest_simple_paths(markov_chain, source, target):
                    if current_cutoff != 1 and len(path) > current_cutoff:
                        break
                    shortest_paths.add(tuple(path))
                    current_cutoff = max(len(path), cutoff+1)
            except nx.NetworkXNoPath:
                continue

    shortest_paths = sorted(list(shortest_paths), key=lambda x: (len(x), *x))

    return shortest_paths


def get_markov_chain_path_probs(markov_chain: nx.DiGraph, all_paths: list[tuple]|None=None, DEBUG: bool = False) -> dict[tuple, float]:
    """
    Compute the probabilities of all unique simple edge paths in a Markov chain.

    Parameters
    ----------
    markov_chain : networkx DiGraph
        The Markov chain represented as a directed graph.
    all_paths : list of tuples, optional
        A list of tuples, each representing a unique simple path in the
        Markov chain. Each path is represented as a tuple of nodes.
        If not given, the function will compute all paths using
        `get_all_paths`.
    DEBUG : bool, optional
        If set to True, performs additional checks on the input data.

    Returns
    -------
    path_probabilities : dict of tuples and floats
        A dictionary where keys are tuples representing unique simple edge paths
        in the Markov chain, and values are the corresponding probabilities.
    """

    if all_paths is None:
        all_paths = get_all_paths(markov_chain, DEBUG)

    path_probabilities = {}
    for path in all_paths:
        if len(path) == 2:

            # check if the start node has a self-loop
            if (path[0], path[0]) not in markov_chain.edges:
                path_probabilities[path] = markov_chain.edges[path]['weight']

            else:
                self_loop_weight = markov_chain.edges[(path[0], path[0])]['weight']
                if DEBUG:
                    if self_loop_weight == 1:
                        raise ValueError("Start node cannot have a self-loop with weight 1.")
                path_probabilities[path] = markov_chain.edges[path]['weight'] / (1 - self_loop_weight)
    
    for path in all_paths:
        if len(path) == 2:
            continue

        # make edges out of the path
        edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        # calculate the probability of the path
        probability = 1
        for edge in edges:
            probability *= path_probabilities[edge]
        path_probabilities[path] = probability

    return path_probabilities


def get_stg_path_probs(all_paths: list[tuple], group_indices: list[list[int]], stg: nx.DiGraph|None = None, transition_matrix: np.ndarray|None = None, DEBUG: bool = False) -> dict[tuple, float]:

    """
    Compute the probability of all paths in a markov chain, using a state transition graph.

    Parameters
    ----------
    all_paths : list of tuples
        A list of tuples, each representing a unique simple path in the markov chain.
    group_indices : list of list of int
        A list of lists, where each list contains the indices of nodes in the same group.
    stg : networkx DiGraph, optional
        The State Transition Graph. If not provided, transition_matrix must be provided.
    transition_matrix : numpy array, optional
        The transition matrix of the STG. If not provided, stg must be provided.
    DEBUG : bool, optional
        If set to True, performs additional checks on the input data.

    Returns
    -------
    path_probabilities : dict of tuples and floats
        A dictionary where keys are tuples representing unique simple edge paths
        in the Markov chain, and values are the corresponding probabilities.
    """
    if transition_matrix is None:
        if stg is None:
            raise ValueError("Either stg or transition_matrix must be provided.")
        else:
            transition_matrix = get_transition_matrix(stg)

    path_transition_matrices = {}
    for path in all_paths:
        # consider edges only
        if len(path) != 2:
            continue

        G1 = group_indices[int(path[0][1:])]
        G2 = group_indices[int(path[1][1:])]

        A = transition_matrix[G1, :][:, G1]
        B = transition_matrix[G1, :][:, G2]

        path_transition_matrices[path] = solve_matrix_equation(A, B, DEBUG=DEBUG)

        if DEBUG:
            if path_transition_matrices[path] is None:
                raise ValueError(f"Failed to compute transition matrix for path {path}.")

    for path in all_paths:
        if len(path) == 2:
            continue

        edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        path_transition_matrix = path_transition_matrices[edges[0]]
        for edge in edges[1:]:
            path_transition_matrix = np.matmul(path_transition_matrix, path_transition_matrices[edge])
        
        path_transition_matrices[path] = path_transition_matrix

    # assume uniform probabilities for initial states
    path_probabilities = {}
    for path in all_paths:
        path_probabilities[path] = np.sum(path_transition_matrices[path])/path_transition_matrices[path].shape[0]

    return path_probabilities
        

def solve_matrix_equation(A: np.ndarray, B: np.ndarray, DEBUG: bool = False) -> np.ndarray:
    """
    Solve a matrix equation of the form X = AX + B

    Parameters
    ----------
    A : numpy array
        Coefficient matrix.
    B : numpy array
        Constant matrix.

    Returns
    -------
    X : numpy array
        Solution matrix.
    """

    if DEBUG:
        if A.shape[0] != A.shape[1]:
            raise ValueError("A must be a square matrix.")

        if A.shape[1] != B.shape[0]:
            raise ValueError("The number of columns in A must be equal to the number of rows in B.")

    # Identity matrix of the same size as A
    I = np.eye(A.shape[0])

    # Calculate (I - A)
    I_minus_A = I - A

    if DEBUG:
        cond_I_minus_A = np.linalg.cond(I_minus_A)
        if cond_I_minus_A > 1000:  # Threshold for ill-conditioning
            print(f"Warning: The matrix (I - A) has a high condition number > 1000: {cond_I_minus_A}")

    # Check the rank of (I - A)
    rank_I_minus_A = np.linalg.matrix_rank(I_minus_A)

    # Check if the rank of (I - A) is less than its size
    if rank_I_minus_A < I_minus_A.shape[0]:
        # If rank is less than the size, the matrix is singular
        # Check if the system is consistent (i.e., there may be infinitely many solutions)
        rank_augmented_matrix = np.linalg.matrix_rank(np.hstack((I_minus_A, B)))
        
        if rank_I_minus_A == rank_augmented_matrix:
            print("The system has infinitely many solutions.")
        else:
            print("The system has no solution.")
        return None
    else:
        # Use np.linalg.lstsq to solve (I - A) X = B in a numerically stable manner
        X, residuals, rank, s = np.linalg.lstsq(I_minus_A, B, rcond=None)
        
        # # Optionally, print residuals for debugging
        # if DEBUG:
        #     print(f"Residuals: {residuals}")
        
        return X


def compare_path_reachability(
    stg_probabilities: dict[tuple, float], 
    markov_probabilities: dict[tuple, float], 
    type: str = "all", 
    DEBUG: bool = False
) -> tuple[int, int, int, int]:
    """
    Calculate the true positives, false positives, true negatives, and false negatives
    between two dictionaries, `stg_probabilities` and `markov_probabilities`, representing
    the probabilities of paths in the Markov chain, obtained using the STG or the Markov chain respectively.

    Parameters
    ----------
    stg_probabilities : dict of tuples and floats
        A dictionary where keys are tuples representing unique simple edge paths
        in the Markov chain, and values are the corresponding probabilities obtained using the STG.
    markov_probabilities : dict of tuples and floats
        A dictionary where keys are tuples representing unique simple edge paths
        in the Markov chain, and values are the corresponding probabilities obtained using the Markov chain.
    type : str, optional
        The type of comparison to perform. Can be one of the following:
        - "all": Compare all paths.
        - "edges": Compare only edges.
        - "non_edges": Compare only non-edges.
        - "path_i": Compare only paths with length i.
    DEBUG : bool, optional
        If set to True, performs additional checks on the input data.

    Returns
    -------
    tuple of int
        A tuple containing four integers: (TP, FP, TN, FN)
        - TP: Number of true positives
        - FP: Number of false positives
        - TN: Number of true negatives
        - FN: Number of false negatives
    """
    if DEBUG:
        for path in stg_probabilities.keys():
            if path not in markov_probabilities:
                raise ValueError(f"Path {path} not found in markov_probabilities.")

        for path in markov_probabilities.keys():
            if path not in stg_probabilities:
                raise ValueError(f"Path {path} not found in stg_probabilities.")

    if type == "all":
        answer = stg_probabilities
        guess = markov_probabilities

    elif type == "edges":
        answer = {}
        guess = {}

        for path in stg_probabilities.keys():
            if len(path) != 2:
                continue

            answer[path] = stg_probabilities[path]
            guess[path] = markov_probabilities[path]

    elif type == "non_edges":
        answer = {}
        guess = {}

        for path in stg_probabilities.keys():
            if len(path) == 2:
                continue

            answer[path] = stg_probabilities[path]
            guess[path] = markov_probabilities[path]
    
    elif type.startswith("path_"):
        path_length = int(type.split("_")[1])
        answer = {}
        guess = {}

        for path in stg_probabilities.keys():
            # len(path) is the number of edges plus 1
            if len(path) != path_length + 1:
                continue

            answer[path] = stg_probabilities[path]
            guess[path] = markov_probabilities[path]

    else:
        raise ValueError("Invalid type.")

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for path in answer.keys():
        if answer[path] != 0 and guess[path] != 0:
            TP += 1
        elif answer[path] == 0 and guess[path] != 0:
            FP += 1
        elif answer[path] != 0 and guess[path] == 0:
            TN += 1
        elif answer[path] == 0 and guess[path] == 0:
            FN += 1

    return TP, FP, TN, FN


def compare_path_rmsd(
    stg_probabilities: dict[tuple, float], 
    markov_probabilities: dict[tuple, float], 
    type: str = "all", 
    DEBUG: bool = False
) -> float:
    """
    Calculate the root mean squared difference (RMSD) between two dictionaries, `stg_probabilities` and `markov_probabilities`, 
    representing the probabilities of path in the Markov chain, obtained using the STG or the markov chain respectively.

    Parameters
    ----------
    stg_probabilities : dict[tuple, float]
        A dictionary where keys are tuples representing unique simple edge paths
        in the Markov chain, and values are the corresponding probabilities obtained using the STG.
    markov_probabilities : dict[tuple, float]
        A dictionary where keys are tuples representing unique simple edge paths
        in the Markov chain, and values are the corresponding probabilities obtained using the Markov chain
    type : str, optional
        The type of comparison to perform. Can be one of the following:

        - "all": Compare all paths.
        - "edges": Compare only edges.
        - "non_edges": Compare only non-edges.
        - "path_i": Compare only paths with length i.

    Returns
    -------
    rmsd : float
        The root mean squared difference between the two dictionaries.
    """
    if DEBUG:
        for path in stg_probabilities.keys():
            if path not in markov_probabilities:
                raise ValueError(f"Path {path} not found in markov_probabilities.")

        for path in markov_probabilities.keys():
            if path not in stg_probabilities:
                raise ValueError(f"Path {path} not found in stg_probabilities.")

    if type == "all":
        answer = stg_probabilities
        guess = markov_probabilities

    elif type == "edges":
        answer = {}
        guess = {}

        for path in stg_probabilities.keys():
            if len(path) != 2:
                continue

            answer[path] = stg_probabilities[path]
            guess[path] = markov_probabilities[path]

    elif type == "non_edges":
        answer = {}
        guess = {}

        for path in stg_probabilities.keys():
            if len(path) == 2:
                continue

            answer[path] = stg_probabilities[path]
            guess[path] = markov_probabilities[path]
    
    elif type.startswith("path_"):
        path_length = int(type.split("_")[1])
        answer = {}
        guess = {}

        for path in stg_probabilities.keys():
            # len(path) is the number of edges plus 1
            if len(path) != path_length + 1:
                continue

            answer[path] = stg_probabilities[path]
            guess[path] = markov_probabilities[path]

    else:
        raise ValueError("Invalid type.")
    
    if len(answer) == 0 or len(guess) == 0:
        return 0

    rmsd = 0
    for path in answer.keys():
        rmsd += (answer[path] - guess[path]) ** 2

    rmsd /= len(answer)

    rmsd = rmsd ** 0.5

    return rmsd

def get_random_path_probs(
    markov_probabilities: dict[tuple, float], 
    seed: int = 0
) -> dict[tuple, float]:
    """
    Generate a dictionary of random path probabilities from a given dictionary of path probabilities.

    Parameters
    ----------
    markov_probabilities : dict of tuples and floats
        A dictionary where keys are tuples representing unique simple edge paths
        in the Markov chain, and values are the corresponding probabilities.
    seed : int, optional
        The seed for the random number generator.

    Returns
    -------
    path_probabilities : dict of tuples and floats
        A dictionary where keys are tuples representing unique simple edge paths
        in the Markov chain, and values are the corresponding probabilities.
    """
    random.seed(seed)

    path_probabilities = {}
    for path in markov_probabilities.keys():
        if len(path) == 2:
            path_probabilities[path] = random.uniform(0, 1)
        else:
            continue

    sum = {}
    for edge in path_probabilities.keys():
        # divide the probability by the sum of probabilities of edges with the same start node
        start_node = edge[0]
        sum[edge] = 0
        for other_edge in path_probabilities.keys():
            if other_edge[0] == start_node:
                sum[edge] += path_probabilities[other_edge]

    for edge in path_probabilities.keys():
        # in case all edges with the same start node have probability 0
        if sum[edge] == 0:
            path_probabilities[edge] = 1
            sum[edge] = 1
        else:
            path_probabilities[edge] /= sum[edge]

    for path in markov_probabilities.keys():
        if len(path) == 2:
            continue

        # make edges out of the path
        edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        # calculate the probability of the path
        probability = 1
        for edge in edges:
            probability *= path_probabilities[edge]
        path_probabilities[path] = probability

    return path_probabilities
