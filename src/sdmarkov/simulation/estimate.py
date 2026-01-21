import numpy as np
import cupy as cp

from pyboolnet.external.bnet2primes import bnet_text2primes
from pyboolnet.prime_implicants import percolate
from pyboolnet.file_exchange import primes2bnet

from sdmarkov.adapter.cubewalkers import (
    CWSingleStep,
    simulate_one_step_aligned_states,
)
from sdmarkov.representation import (
    partial_assignments_to_masks,
    decompose_subcubes,
)
from sdmarkov.simulation.sample import (
    prepare_group_sampler,
    sample_walkers_from_group,
)
from sdmarkov.succession_diagram import build_sd_trap_spaces
from sdmarkov.simulation.classify import classify_walkers


def estimate_sd_transition_matrix(
    rules: str,
    n_walkers: int,
    xp=cp,
    return_aux=False,
    threads_per_block=(16,16),
):
    """
    Estimate the SD-grouped transition matrix via Monte Carlo simulation.

    Parameters
    ----------
    rules : str
        Boolean network rules (cubewalkers / pyboolnet compatible).
    n_walkers : int
        Number of walkers per source group.
    xp : module
        Backend array module (default: cupy).
    return_aux : bool
        If True, also return auxiliary objects (sampler, nodes, trap spaces).
    threads_per_block : tuple[int, int], optional
        How many threads should be in each block for each dimension of the N
        x W array, by default `(16, 16)`. See CUDA documentation for details.
        
    Returns
    -------
    T_empirical : np.ndarray
        Empirical SD-grouped transition matrix, shape (G, G).
    aux : dict (optional)
        Intermediate objects useful for debugging or inspection.
    """
    # --- Parse Boolean network ---
    primes = bnet_text2primes(rules)

    # --- Percolate constants ---
    percolated_primes = percolate(primes, remove_constants=True, copy=True)
    if len(percolated_primes) == 0:
        if return_aux:
            return np.zeros((0, 0), dtype=float), {
                "sampler": None,
                "trap_spaces": [],
                "canonical_subcubes": [],
                "nodes": [],
            }
        return np.zeros((0, 0), dtype=float)

    percolated_bnet = primes2bnet(percolated_primes)

    # --- 1. Build SD trap spaces ---
    ts_nodes, trap_spaces = build_sd_trap_spaces(percolated_bnet)

    # --- 2. Convert trap spaces to canonical subcubes ---
    masks, values = partial_assignments_to_masks(trap_spaces, ts_nodes)
    trap_spaces_masks = [
        (masks[:, i], values[:, i], f"T{i}")
        for i in range(masks.shape[1])
    ]
    canonical = decompose_subcubes(trap_spaces_masks)

    # --- 3. Prepare group sampler ---
    sampler = prepare_group_sampler(canonical, xp=xp)

    n_groups = len(sampler.group_name_to_id)

    # --- 4. Cubewalkers single-step model ---
    cw_step = CWSingleStep(
        rules=percolated_bnet,
        n_walkers=n_walkers,
    )

    # --- 5. Estimate transitions ---
    T_empirical = np.zeros((n_groups, n_groups), dtype=float)

    for group_name, i in sampler.group_name_to_id.items():
        # Sample walkers from source group
        states = sample_walkers_from_group(
            sampler,
            target_group=group_name,
            n_walkers=n_walkers,
            xp=xp,
        )

        # One asynchronous step
        states_next = simulate_one_step_aligned_states(
            states=states,
            sampler_nodes=ts_nodes,
            model=cw_step,
            threads_per_block=threads_per_block,
        )

        # Classify destination groups
        labels = classify_walkers(
            states_next,
            sampler,
            xp=xp,
        )

        counts = np.bincount(
            labels.get(),
            minlength=n_groups,
        )
        T_empirical[i] = counts / counts.sum()

    if return_aux:
        return T_empirical, {
            "sampler": sampler,
            "trap_spaces": trap_spaces,
            "canonical_subcubes": canonical,
            "nodes": ts_nodes,
        }

    return T_empirical
