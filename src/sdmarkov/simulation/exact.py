"""
Exact construction of SD-grouped transition matrices
via full state-space enumeration.

This module provides ground-truth transition operators
used for validation and benchmarking of Monte Carlo
estimators.
"""

import numpy as np

from pyboolnet.external.bnet2primes import bnet_text2primes
from pyboolnet.prime_implicants import percolate
from pyboolnet.file_exchange import primes2bnet
from pyboolnet.state_transition_graphs import primes2stg

from sdmarkov.transition_matrix import get_transition_matrix
from sdmarkov.matrix_operations import compress_matrix
from sdmarkov.grouping import sd_grouping


def exact_sd_transition_matrix_from_rules(
    rules: str,
    asynchronous: bool = True,
):
    """
    Compute the exact SD-grouped transition matrix from Boolean rules.

    This function:
      1. Builds the full asynchronous state transition graph
      2. Computes the exact state-level transition matrix
      3. Groups states using SD grouping
      4. Compresses to a group-level transition matrix

    Parameters
    ----------
    rules : str
        Boolean network rules in bnet format.
    asynchronous : bool
        If True, use asynchronous update scheme (default).

    Returns
    -------
    T_sd : np.ndarray
        Exact SD-grouped transition matrix.
    sd_indices : list[int]
        Mapping from full states to SD groups.
    """
    # --- Parse Boolean network ---
    primes = bnet_text2primes(rules)

    # --- Percolate constants ---
    percolated_primes = percolate(primes, remove_constants=True, copy=True)

    if len(percolated_primes) == 0:
        return np.zeros((0, 0), dtype=float), []

    percolated_bnet = primes2bnet(percolated_primes)

    # --- State transition graph ---
    update_mode = "asynchronous" if asynchronous else "synchronous"
    stg = primes2stg(percolated_primes, update_mode)

    # --- Full transition matrix ---
    T_full = get_transition_matrix(stg)

    # --- SD grouping ---
    sd_indices = sd_grouping(percolated_bnet)

    # --- Compress ---
    T_sd = compress_matrix(T_full, sd_indices)

    return T_sd, sd_indices
