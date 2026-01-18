"""
sdmarkov.adapter.cubewalkers
-----------------------------

Adapter layer for running single-step asynchronous simulations
with pre-sampled states using cubewalkers.Model.

This module provides:

1. CWSingleStep - minimal wrapper around cubewalkers.Model for one-step asynchronous updates.
2. simulate_one_step_aligned_states - canonical SDMarkov function to:
   - align sampled states to the cubewalkers model node order
   - run one asynchronous update
   - realign the result back to the sampler node order
"""

import numpy as np
import cupy as cp
import cubewalkers as cw


class CWSingleStep:
    """
    Minimal wrapper for single-step asynchronous simulation with custom initial states.
    Only works with CuPy arrays internally.
    """

    def __init__(self, rules: str, n_walkers: int):
        """
        Parameters
        ----------
        rules : str
            Boolean rules to define the network, compatible with cubewalkers.Model.
        n_walkers : int
            Number of ensemble walkers to simulate.
        """
        self.model = cw.Model(rules, n_time_steps=1, n_walkers=n_walkers)
        self.n_walkers = n_walkers

    def set_initial_states(self, states: np.ndarray | cp.ndarray):
        """
        Overwrite the model's initial states with a pre-sampled state array.

        Parameters
        ----------
        states : np.ndarray or cp.ndarray
            Shape (n_variables, n_walkers)
        """
        # Convert to CuPy if necessary
        if not isinstance(states, cp.ndarray):
            states = cp.array(states)

        if states.shape != (self.model.n_variables, self.n_walkers):
            raise ValueError(
                f"Expected shape ({self.model.n_variables}, {self.n_walkers}), got {states.shape}"
            )

        self.model.initial_states = states

    def simulate_step(self) -> cp.ndarray:
        """
        Run a single asynchronous update.

        Returns
        -------
        traj : cp.ndarray
            Shape (1, n_variables, n_walkers)
        """
        self.model.simulate_ensemble(
            T_window=1,
            averages_only=False,
            maskfunction=cw.update_schemes.asynchronous
        )
        return self.model.trajectories


def simulate_one_step_aligned_states(
    states: np.ndarray | cp.ndarray,
    sampler_nodes: list[str],
    model: CWSingleStep
) -> np.ndarray | cp.ndarray:
    """
    Run a single-step simulation on pre-sampled states.

    Aligns the input states to the model node order, runs one asynchronous
    update, and realigns the result back to the original sampler node order.

    Parameters
    ----------
    states : np.ndarray or cp.ndarray
        Pre-sampled states, shape (n_variables, n_walkers)
    sampler_nodes : list[str]
        Node names corresponding to rows of `states`
    model : CWSingleStep
        Wrapper instance already initialized with cubewalkers.Model

    Returns
    -------
    updated_states_aligned : np.ndarray or cp.ndarray
        Updated states after one asynchronous step,
        shape (n_variables, n_walkers),
        same type as input `states`.
    """
    input_is_cupy = isinstance(states, cp.ndarray)
    # Convert to CuPy for cubewalkers
    states_cp = states if input_is_cupy else cp.array(states)

    # Align states to model nodes
    model_nodes = model.model.varnames
    perm_forward = [sampler_nodes.index(n) for n in model_nodes]
    states_aligned = states_cp[perm_forward, :]

    # Inject and simulate
    model.set_initial_states(states_aligned)
    traj = model.simulate_step()  # shape (1, n_variables, n_walkers)
    updated_states = traj[0, :, :]  # drop time dimension

    # Realign back to sampler node order
    perm_reverse = np.argsort(perm_forward)
    updated_states_aligned = updated_states[perm_reverse, :]

    # Convert back to original type
    if not input_is_cupy:
        updated_states_aligned = cp.asnumpy(updated_states_aligned)

    return updated_states_aligned
