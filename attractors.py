import numpy as np

from transition_matrix import check_transition_matrix
from graph import get_stg, get_markov_chain
from scc_dags import get_scc_dag, get_attractor_states


def attractor_or_transient(T_inf, DEBUG=False):
    """
    Determine if each state is part of an attractor or transient state.

    Parameters
    ----------
    T_inf : np.ndarray
        The transition matrix at t=inf.
    DEBUG : bool, optional
        If True, performs additional checks.

    Returns
    -------
    np.ndarray
        A matrix of 0s and 1s, where the ith element is 1 if state i is part of an
        attractor, and 0 otherwise.
    """
    if DEBUG:
        check_transition_matrix(T_inf)

    attractor_matrix = np.zeros((T_inf.shape[0], 1))

    for i in range(T_inf.shape[0]):
        if T_inf[i][i] != 0:
            attractor_matrix[i][0] = 1

    return attractor_matrix


def get_predicted_attractors(transition_matrix, group_indices=None, as_indices=False, DEBUG=False):
    """
    Get the states of the attractors predicted by the transition matrix.

    Parameters
    ----------
    transition_matrix : np.ndarray
        The transition matrix.
    group_indices : list[list[int]], optional
        A list of sublists, where each sublist contains the indices of nodes in the same group.
        Must be provided if the transition matrix is grouped.
    as_indices : bool, optional
        If True, returns the indices of the attractors.
    DEBUG : bool, optional
        If True, performs additional checks.

    Returns
    -------
    attractor_states : list[list[int]]
        The states of the attractors predicted by the transition matrix.
    """

    if DEBUG:
        check_transition_matrix(transition_matrix, compressed=True)

    if group_indices is None:
        markov_chain = get_stg(transition_matrix, DEBUG=DEBUG)
    else:
        markov_chain = get_markov_chain(transition_matrix, group_indices, DEBUG=DEBUG)

    scc_dag = get_scc_dag(markov_chain, DEBUG=DEBUG)
    attractor_states = get_attractor_states(scc_dag, as_indices=as_indices, DEBUG=DEBUG)

    return attractor_states