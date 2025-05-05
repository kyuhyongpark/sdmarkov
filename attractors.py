import numpy as np

from transition_matrix import check_transition_matrix


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