def _classify_full(states, sampler, xp):
    """
    Full classification against all canonical subcubes.
    states: (N, W')
    returns: (W',) subcube indices
    """

    # (N, K, W')
    matches = (states[:, None, :] & sampler.masks[:, :, None]) == sampler.values[:, :, None]

    # (K, W')
    matches = xp.all(matches, axis=0)

    # assume exactly one match per walker
    return xp.argmax(matches, axis=0)


def classify_walkers(states, sampler, xp, prev_subcube_idx=None):
    """
    Classify walkers into canonical subcubes.

    Parameters
    ----------
    states : (N, W) bool
    sampler : object with masks, values, group_ids
    xp : numpy or cupy
    prev_subcube_idx : (W,) int or None

    Returns
    -------
    group_ids : (W,) int
    subcube_idx : (W,) int
    """

    N, W = states.shape

    # ----------------------------
    # Fast path: validate previous subcube
    # ----------------------------
    if prev_subcube_idx is not None:
        mask = sampler.masks[:, prev_subcube_idx]      # (N, W)
        value = sampler.values[:, prev_subcube_idx]    # (N, W)

        still_valid = xp.all((states & mask) == value, axis=0)

        if xp.all(still_valid):
            return (
                sampler.group_ids[prev_subcube_idx],
                prev_subcube_idx,
            )

        # split walkers
        valid_idx = xp.where(still_valid)[0]
        invalid_idx = xp.where(~still_valid)[0]

        subcube_idx = xp.empty(W, dtype=prev_subcube_idx.dtype)
        subcube_idx[valid_idx] = prev_subcube_idx[valid_idx]

        # classify only invalid walkers
        subcube_idx[invalid_idx] = _classify_full(
            states[:, invalid_idx], sampler, xp
        )

        return (
            sampler.group_ids[subcube_idx],
            subcube_idx,
        )

    # ----------------------------
    # Slow path: full classification
    # ----------------------------
    subcube_idx = _classify_full(states, sampler, xp)

    return (
        sampler.group_ids[subcube_idx],
        subcube_idx,
    )
