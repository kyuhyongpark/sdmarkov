def classify_walkers(states, sampler, xp, prev_subcube_idx=None):
    """
    Classify walkers with fast-path using previous subcube index.

    Args:
        states: (N, W) boolean array
        sampler: GroupSampler
        xp: np or cp
        prev_subcube_idx: (W,) int array of previous k, or None

    Returns:
        group_ids: (W,) int array
        subcube_idx: (W,) int array
    """
    N, W = states.shape
    K = sampler.masks.shape[1]

    group_ids = xp.full(W, -1, dtype=int)
    subcube_idx = xp.full(W, -1, dtype=int)
    active = xp.ones(W, dtype=bool)

    # --- Fast path: check previous subcube first ---
    if prev_subcube_idx is not None:
        for k in xp.unique(prev_subcube_idx):
            idx = xp.where((prev_subcube_idx == k) & active)[0]
            if idx.size == 0:
                continue

            mask_k = sampler.masks[:, k][:, None]
            value_k = sampler.values[:, k][:, None]

            hit = xp.all((states[:, idx] & mask_k) == value_k, axis=0)

            if hit.any():
                matched = idx[hit]
                group_ids[matched] = sampler.group_ids[k]
                subcube_idx[matched] = k
                active[matched] = False

    # --- Fallback: normal early-exit scan ---
    for k in range(K):
        if not active.any():
            break

        mask_k = sampler.masks[:, k][:, None]
        value_k = sampler.values[:, k][:, None]

        idx = xp.where(active)[0]
        hit = xp.all((states[:, idx] & mask_k) == value_k, axis=0)

        if hit.any():
            matched = idx[hit]
            group_ids[matched] = sampler.group_ids[k]
            subcube_idx[matched] = k
            active[matched] = False

    assert xp.all(group_ids >= 0), "Semantic invariant violated"

    return group_ids, subcube_idx
