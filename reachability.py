import numpy as np

from transition_matrix import check_transition_matrix
from matrix_operations import get_block_triangular
from helper import indices_to_states


def get_convergence_matrix(
    T_inf: np.ndarray, 
    scc_indices: list[list[int]], 
    attractor_indices: list[list[int]], 
    DEBUG: bool = False
) -> tuple[np.ndarray, list[str], list[list[str]]]:
    """
    Compute the convergence matrix that represents the probability of reaching each attractor from each transient state.

    Parameters
    ----------
    T_inf : np.ndarray
        The transition matrix at t=inf.
    scc_indices : list of list of int
        The indices of the strongly connected components in the transition matrix.
    attractor_indices : list of list of int
        The indices of the attractor states in the transition matrix.
    DEBUG : bool, optional
        If True, performs additional checks.

    Returns
    -------
    tuple
        A tuple containing:
        - convergence_matrix : np.ndarray
            A matrix representing the probability of reaching each attractor from each transient state.
        - transient_states : list of str
            A list of transient states.
        - attractor_states : list of list of str
            A list of attractor states.
    """

    if DEBUG:
        # Check that the given matrix is a transition matrix
        check_transition_matrix(T_inf)
    
        # Check that the scc indices are of the right size
        if len(set([index for scc in scc_indices for index in scc])) != T_inf.shape[0]:
            raise ValueError("The scc indices must be of the right size.")

        # Check that the scc indices are valid
        for scc in scc_indices:
            for state in scc:
                if 0 > state or state >= T_inf.shape[0]:
                    raise ValueError("The scc indices must be valid.")

        # Check that the scc indices are mutually exclusive
        for i in range(len(scc_indices)):
            for j in range(i + 1, len(scc_indices)):
                if set(scc_indices[i]).intersection(set(scc_indices[j])):
                    raise ValueError("The scc indices must be mutually exclusive.")

        # Check that the attractor indices are valid
        for attractor in attractor_indices:
            for state in attractor:
                if 0 > state or state >= T_inf.shape[0]:
                    raise ValueError("The attractor indices must be valid.")

        # Check that the attractor indices are mutually exclusive
        for i in range(len(attractor_indices)):
            for j in range(i + 1, len(attractor_indices)):
                if set(attractor_indices[i]).intersection(set(attractor_indices[j])):
                    raise ValueError("The attractor indices must be mutually exclusive.")
                
        # Check that the attractor indices are present in the scc indices
        for attractor in attractor_indices:
            if attractor not in scc_indices:
                raise ValueError("The attractor indices must be present in the scc indices.")

    N = int(np.log2(T_inf.shape[0]))

    # Get the attractor states
    attractor_states = indices_to_states(attractor_indices, N, DEBUG=DEBUG)

    # Get the transient states
    transient_index_groups = [group for group in scc_indices if group not in attractor_indices]
    if len(transient_index_groups) == 0:
        # If there are no transient states, return an empty matrix
        return np.zeros((0, 0)), [], attractor_states
    transient_state_groups = indices_to_states(transient_index_groups, N, DEBUG=DEBUG)
    transient_states = [state for group in transient_state_groups for state in group]

    # Make the matrix block diagonal
    block_T_inf, _ = get_block_triangular(T_inf, scc_indices=scc_indices, DEBUG=DEBUG)

    # Get the basin region of the transition matrix that corresponds to transitions from transient states to attractor states
    # The basin region is the top right corner of the block triangular matrix    
    all_attractor_indices = []
    for attractor in attractor_indices:
        all_attractor_indices.extend(attractor)

    basin_region = block_T_inf[:-len(all_attractor_indices), -len(all_attractor_indices):]

    # Sum up the columns of the basin region to get transition probabilities from transient states to attractors
    attractor_sizes = [len(attractor) for attractor in attractor_indices]
    start = 0
    summed_columns = []

    for size in attractor_sizes:
        end = start + size
        summed = basin_region[:, start:end].sum(axis=1, keepdims=True)
        summed_columns.append(summed)
        start = end

    convergence_matrix = np.hstack(summed_columns)

    # Normalize the rows of the convergence matrix if the sum of each row is greater than 1.
    # Note that the sum of each row may be greater than 1 due to floating point errors.
    # Also note that the sum of each row may be less than 1 due to overestimating attractor states, which should not be corrected.
    for i in range(convergence_matrix.shape[0]):
        if np.sum(convergence_matrix[i, :]) > 1:
            convergence_matrix[i, :] /= np.sum(convergence_matrix[i, :])


    if DEBUG:
        if convergence_matrix.shape[0] != len(transient_states):
            raise ValueError("The number of transient states does not match the number of rows in the convergence matrix.")
        if convergence_matrix.shape[1] != len(attractor_states):
            raise ValueError("The number of attractor states does not match the number of columns in the convergence matrix.")

    return convergence_matrix, transient_states, attractor_states


