def classify_walkers(states, sampler, xp):
    """
    Classify each walker into its canonical subcube / group.

    Args:
        states: (N, W) boolean array
        sampler: GroupSampler object
        xp: np or cp (NumPy or CuPy)

    Returns:
        group_ids: (W,) int array
    """
    N, W = states.shape
    K = sampler.masks.shape[1]  # number of canonical subcubes

    # --- Fully vectorized version (default) ---
    states_exp = states[:, None, :]           # (N, 1, W)
    masks_exp = sampler.masks[:, :, None]    # (N, K, 1)
    values_exp = sampler.values[:, :, None]  # (N, K, 1)

    matches = (states_exp & masks_exp) == values_exp  # (N, K, W)
    matches = matches.all(axis=0)                     # (K, W)

    # --- Memory-efficient alternative (loop over subcubes) ---
    # matches = xp.zeros((K, W), bool)
    # for k in range(K):
    #     mask_k = sampler.masks[:, k][:, None]   # (N, 1)
    #     value_k = sampler.values[:, k][:, None] # (N, 1)
    #     matches[k] = xp.all((states & mask_k) == value_k, axis=0)  # (W,)
    
    # Semantic invariant: exactly one match per walker
    assert xp.all(matches.sum(axis=0) == 1), "Semantic invariant violated"

    # Map subcube index to group id
    subcube_to_group = sampler.group_ids           # (K,)
    subcube_indices = xp.argmax(matches, axis=0)  # (W,)
    group_ids = subcube_to_group[subcube_indices]  # (W,)

    return group_ids
