from dataclasses import dataclass
import math

Mask = object   # (N,) bool array
Value = object  # (N,) bool array
Subcube = tuple # (mask, value, group_name)


@dataclass(frozen=True)
class GroupSampler:
    # canonical definition
    masks: object        # (N, K), bool
    values: object       # (N, K), bool
    group_ids: object    # (K,), int

    # subcube metadata
    free_bits: object    # (K,), int
    log_sizes: object    # (K,), float

    # grouping structure
    groups: dict         # group_id -> (K_g,) int array
    group_log_sizes: dict  # group_id -> float

    group_name_to_id: dict
    id_to_group_name: list

def prepare_group_sampler(
    canonical_subcubes: list[Subcube],
    *,
    xp,
) -> GroupSampler:
    """
    Prepare GroupSampler from canonical subcubes produced by decompose_subcubes.

    Assumes canonical_subcubes are:
    - disjoint
    - complete
    - each assigned to exactly one trap space (group_name)
    """

    K = len(canonical_subcubes)
    if K == 0:
        raise ValueError("No canonical subcubes provided")

    group_order = {group: i for i, (_, _, group) in enumerate(canonical_subcubes)}

    masks_list, values_list, group_names = zip(*canonical_subcubes)

    masks = xp.stack([xp.asarray(m, dtype=bool) for m in masks_list], axis=1)
    values = xp.stack([xp.asarray(v, dtype=bool) for v in values_list], axis=1)


    group_name_to_id = {}
    id_to_group_name = []
    group_ids_py = []

    # Get unique group names but preserve original ordering
    unique_names = sorted(set(group_names),key=lambda g: group_order[g])

    group_name_to_id = {name: i for i, name in enumerate(unique_names)}
    id_to_group_name = list(unique_names)
    group_ids_py = [group_name_to_id[g] for g in group_names]

    group_ids = xp.asarray(group_ids_py, int)

    free_bits = (~masks).sum(axis=0)
    log_sizes = free_bits * math.log(2.0)

    groups = {}
    group_log_sizes = {}

    for k, g in enumerate(group_ids_py):
        groups.setdefault(g, []).append(k)

    for g, ks in groups.items():
        ks = xp.asarray(ks, int)
        groups[g] = ks
        group_log_sizes[g] = xp.log(xp.exp(log_sizes[ks]).sum())

    return GroupSampler(
        masks=masks,
        values=values,
        group_ids=group_ids,
        free_bits=free_bits,
        log_sizes=log_sizes,
        groups=groups,
        group_log_sizes=group_log_sizes,
        group_name_to_id=group_name_to_id,
        id_to_group_name=id_to_group_name,
    )


def sample_walkers_from_group(sampler, target_group, n_walkers, xp):
    # resolve group
    if isinstance(target_group, str):
        try:
            g = sampler.group_name_to_id[target_group]
        except KeyError:
            raise KeyError(f"Unknown group name: {target_group}")
    else:
        g = int(target_group)

    ks = sampler.groups[g]
    logw = sampler.log_sizes[ks]

    log_weights = logw - xp.max(logw)
    probs = xp.exp(log_weights)
    probs /= probs.sum()

    subcube_idx = xp.random.choice(ks, size=n_walkers, p=probs)  # (W,)

    N = sampler.masks.shape[0]
    W = n_walkers

    # random free bits
    states = xp.random.randint(0, 2, size=(N, W), dtype=xp.int8).astype(bool)

    mask_matrix = sampler.masks[:, subcube_idx]    # (N, W)
    value_matrix = sampler.values[:, subcube_idx]  # (N, W)

    states = (states & ~mask_matrix) | (value_matrix & mask_matrix)

    return states, subcube_idx

