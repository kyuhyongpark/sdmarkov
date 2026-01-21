def classify_walkers(states, sampler, xp):
    """
    Classify each walker into its canonical subcube / group
    using early exit to reduce work and memory.

    Args:
        states: (N, W) boolean array
        sampler: GroupSampler object
        xp: np or cp

    Returns:
        group_ids: (W,) int array
    """
    N, W = states.shape
    K = sampler.masks.shape[1]

    group_ids = xp.full(W, -1, dtype=int)
    active = xp.ones(W, dtype=bool)   # walkers not yet classified

    for k in range(K):
        if not active.any():
            break

        mask_k = sampler.masks[:, k][:, None]     # (N, 1)
        value_k = sampler.values[:, k][:, None]   # (N, 1)

        states_a = states[:, active]               # (N, Wa)
        hit = xp.all((states_a & mask_k) == value_k, axis=0)  # (Wa,)

        if hit.any():
            idx = xp.nonzero(active)[0][hit]
            group_ids[idx] = sampler.group_ids[k]
            active[idx] = False

    # Semantic invariant: exactly one match per walker
    assert xp.all(group_ids >= 0), "Semantic invariant violated"

    return group_ids
