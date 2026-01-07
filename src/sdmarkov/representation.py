import warnings

import numpy as np
from typing import List, Tuple

from sdmarkov.succession_diagram import validate_no_missing_intersections

# Type aliases for clarity
Mask = np.ndarray  # shape (N,), bool
Value = np.ndarray  # shape (N,), bool
Subcube = Tuple[Mask, Value, str]  # (mask, value, group_name)
Masks = np.ndarray  # shape (N, K), bool
Values = np.ndarray  # shape (N, K), bool


def states_to_indices(state_groups: list[list[str]], DEBUG: bool = False) -> list[list[int]]:
    """
    Convert groups of binary strings to groups of indices.

    Parameters
    ----------
    state_groups: list[list[str]]
        A list of sublists, where each sublist contains binary strings that represent states.
    DEBUG: bool, optional
        If True, checks if all state strings have the same length and are mutually exclusive.

    Returns
    -------
    list[list[int]]
        A list of sublists, where each sublist contains indices that correspond to the states.
    
    Examples
    --------
    >>> states_to_indices([['0100', '0101', '0110', '0111', '1101', '1111'], ['1001', '1011'], ['0000', '0001', '0010'], ['0011'], ['1100', '1110'], ['1000', '1010']])
    [[4, 5, 6, 7, 13, 15], [9, 11], [0, 1, 2], [3], [12, 14], [8, 10]]
    """

    if DEBUG:
        # Check if all state strings are of the same length
        lengths = [len(state) for state_group in state_groups for state in state_group]
        if len(set(lengths)) != 1:
            raise ValueError("Not all states have the same length")
        
        # Check if all groups are mutually exclusive
        all_states = [state for state_group in state_groups for state in state_group]
        if len(all_states) != len(set(all_states)):
            raise ValueError("States are not mutually exclusive (duplicates found)")

    # Convert binary strings to integers
    index_groups = [[int(state, 2) for state in state_group] for state_group in state_groups]
    
    return index_groups


def indices_to_states(index_groups: list[list[int]], N: int, DEBUG: bool = False) -> list[list[str]]:
    """
    Convert groups of indices to groups of states.

    Parameters
    ----------
    index_groups: list[list[int]]
        A list of sublists, where each sublist contains indices that correspond to the states.
    N: int
        The number of nodes in a state.
    DEBUG: bool, optional
        If True, checks if all groups are mutually exclusive.

    Returns
    -------
    list[list[str]]
        A list of sublists, where each sublist contains binary strings that correspond to the indices.
    
    Examples
    --------
    >>> indices_to_states([[4, 5, 6, 7, 13, 15], [9, 11], [0, 1, 2], [3], [12, 14], [8, 10]], 4)
    [['0100', '0101', '0110', '0111', '1101', '1111'], ['1001', '1011'], ['0000', '0001', '0010'], ['0011'], ['1100', '1110'], ['1000', '1010']]
    """

    if DEBUG:
        # Check if N is an integer
        if not isinstance(N, int):
            raise ValueError(f"N must be an integer: {N=}")

        # Check if N is large enough
        largest_index = max([max(index_group) for index_group in index_groups if index_group])
        if largest_index >= 2**N:
            raise ValueError("N is too small")

        # Check if all groups are mutually exclusive
        all_indices = [index for index_group in index_groups for index in index_group]
        if len(all_indices) != len(set(all_indices)):
            raise ValueError("Indices are not mutually exclusive (duplicates found)")

    # Convert integers to binary strings
    state_groups = [[f"{index:0{N}b}" for index in index_group] for index_group in index_groups]
    
    return state_groups


def partial_assignment_to_mask(
    assignment: dict[str, int],
    nodes: list[str],
) -> tuple[Mask, Value]:
    """
    Convert a single partial assignment into mask/value vectors.

    Parameters
    ----------
    assignment
        Mapping from node name to 0 or 1. Missing nodes are unconstrained.
    nodes
        Fixed ordering of all variables.

    Returns
    -------
    mask : Mask
        Boolean array of shape (N,). True where the variable is constrained.
    value : Value
        Boolean array of shape (N,). Value is meaningful only where mask is True.
        But it must be set to False where mask is False.
    
    Notes
    -----
    The returned (mask, value) pair satisfies the invariant that
    value[i] == False whenever mask[i] == False,
    so that membership can be tested via (state & mask) == value.
    """

    N = len(nodes)

    mask = np.zeros(N, dtype=bool)
    value = np.zeros(N, dtype=bool)

    for i, node in enumerate(nodes):
        if node in assignment:
            mask[i] = True
            value[i] = bool(assignment[node])

    return mask, value


def partial_assignments_to_masks(
    assignments: list[dict[str, int]],
    nodes: list[str],
) -> tuple[Masks, Values]:
    """
    Convert a list of partial assignments into mask/value matrices.

    Parameters
    ----------
    assignments
        List of partial assignments.
    nodes
        Fixed ordering of all variables.

    Returns
    -------
    mask : Masks
        Boolean array of shape (N, K). True where the variable is constrained.
    value : Values
        Boolean array of shape (N, K). Value is meaningful only where mask is True.
        But it must be set to False where mask is False.
    """
    N = len(nodes)
    K = len(assignments)

    mask = np.zeros((N, K), dtype=bool)
    value = np.zeros((N, K), dtype=bool)

    for k, assignment in enumerate(assignments):
        m, v = partial_assignment_to_mask(assignment, nodes)
        mask[:, k] = m
        value[:, k] = v

    return mask, value


def mask_to_partial_assignment(
    mask: np.ndarray,
    value: np.ndarray,
    nodes: list[str] | None = None
) -> dict[str, int]:
    """
    Convert a mask/value pair to a partial assignment dictionary.

    Parameters
    ----------
    mask : np.ndarray, shape (N,), bool
        Boolean mask indicating which bits are constrained.
    value : np.ndarray, shape (N,), bool
        Values for constrained bits. Value is ignored where mask is False.
    nodes : list[str], optional
        Names of nodes. If None, defaults to ["0", "1", "2", ..., "N-1"].

    Returns
    -------
    dict[str, int]
        Mapping from node name to 0 or 1 for all constrained bits.
    """
    N = len(mask)
    if nodes is None:
        nodes = [str(i) for i in range(N)]
    if len(nodes) != N:
        raise ValueError(f"Length of nodes ({len(nodes)}) must match length of mask ({N}).")

    assignment = {nodes[i]: int(value[i]) for i in range(N) if mask[i]}
    return assignment


def mask_value_to_pattern(
    mask: Mask,
    value: Value,
) -> str:
    """
    Convert a (mask, value) pair into a human-readable pattern string.

    The pattern uses:
    - '*' for unconstrained bits (mask == 0)
    - '0' for constrained zeros (mask == 1, value == 0)
    - '1' for constrained ones  (mask == 1, value == 1)

    Examples
    --------
    mask  = [1, 0, 1, 0]
    value = [0, 0, 1, 0]
    -> '0*1*'
    """
    chars = []
    for m, v in zip(mask, value):
        if not m:
            chars.append("*")
        else:
            chars.append("1" if v else "0")
    return "".join(chars)


def subtract_subcube(
    A: tuple[Mask, Value],
    B: tuple[Mask, Value],
) -> list[tuple[Mask, Value]]:
    """
    Return a disjoint decomposition of A \\ B.

    Parameters
    ----------
    A : (mask, value)
        The minuend subcube.
    B : (mask, value)
        The subtrahend subcube.

    Returns
    -------
    list of (mask, value)
        Pairwise disjoint subcubes whose union equals A \\ B.

    Notes
    -----
    - If A and B do not overlap, returns [A].
    - If B fully covers A, returns [].
    """

    mA, vA = A
    mB, vB = B
    N = len(mA)

    # Check for incompatibility on fixed bits: no overlap
    for i in range(N):
        if mA[i] and mB[i] and vA[i] != vB[i]:
            return [(mA.copy(), vA.copy())]

    # Indices where A is free but B is fixed
    free_but_fixed = [
        i for i in range(N)
        if not mA[i] and mB[i]
    ]

    residual = []

    for k, i in enumerate(free_but_fixed):
        m = mA.copy()
        v = vA.copy()

        # Match B on earlier bits
        for j in free_but_fixed[:k]:
            m[j] = True
            v[j] = vB[j]

        # Deviate from B at bit i
        m[i] = True
        v[i] = not vB[i]

        residual.append((m, v))

    return residual


def decompose_subcubes(
    subcubes: List[Subcube],
    DEBUG: bool = False,
) -> List[Subcube]:
    """
    Decompose a list of subcubes into a deterministic, disjoint set of subcubes.

    Parameters
    ----------
    subcubes : list of (mask, value, group)
        Subcubes that may overlap.
        The input is expected to be **closed under intersection** for the
        decomposition to be well-defined.
    DEBUG : bool, optional
        If True, checks whether the input subcubes are intersection-closed and
        prints a warning if missing intersections are found.

    Returns
    -------
    canonical_subcubes : list of (mask, value, group)
        Disjoint subcubes whose union equals the union of the input subcubes.

    Notes
    -----
    - The decomposition is not mathematically unique.
      This function enforces determinism by internally sorting subcubes
      using a fixed tie-breaking rule.
    - The result is independent of the input order, but depends on the
      chosen internal ordering.
    - The function does not alter the semantics of the input subcubes;
      it only partitions overlapping regions.
    """

    # ----------------- DEBUG: check intersection closure -----------------
    if DEBUG:
        N = len(subcubes[0][0])
        assignments = [
            mask_to_partial_assignment(mask, value)
            for mask, value, _ in subcubes
        ]
        if not validate_no_missing_intersections(assignments, DEBUG=True):
            warnings.warn(
                "Input subcubes are not closed under intersection. ",
                RuntimeWarning
            )

    # ----------------- Main decomposition logic -----------------
    # - Specificity ordering is required for correctness
    # - Remaining keys exist only to make the result deterministic
    def subcube_sort_key(subcube):
        mask, value, _ = subcube
        return (
            -int(mask.sum()),           # semantic requirement
            tuple(mask.astype(int)),    # deterministic geometry
            tuple(value.astype(int)),
        )

    sorted_subcubes = sorted(subcubes, key=subcube_sort_key)

    canonical: List[Subcube] = []

    for mask, value, group in sorted_subcubes:
        pending: List[tuple[Mask, Value]] = [(mask.copy(), value.copy())]

        for cm, cv, _ in canonical:
            new_pending: List[tuple[Mask, Value]] = []
            for m, v in pending:
                new_pending.extend(subtract_subcube((m, v), (cm, cv)))
            pending = new_pending

        for m, v in pending:
            canonical.append((m, v, group))

    return canonical
