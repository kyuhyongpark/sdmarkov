import pickle
import numpy as np
from dataclasses import dataclass
from pathlib import Path

from pyboolnet.external.bnet2primes import bnet_text2primes
from pyboolnet.file_exchange import primes2bnet
from pyboolnet.prime_implicants import percolate
from pyboolnet.state_transition_graphs import primes2stg
from pyboolnet.trap_spaces import compute_trap_spaces

from sdmarkov.attractors import get_predicted_attractors
from sdmarkov.grouping import sd_grouping, null_grouping, random_grouping
from sdmarkov.matrix_operations import nsquare, compress_matrix, expand_matrix
from sdmarkov.scc_dags import get_scc_dag, get_attractor_states
from sdmarkov.transition_matrix import get_transition_matrix


@dataclass
class RandomContext:
    n_random: int
    seeds: list

    # random groupings and matrices
    indices: list
    Trandom: list
    Trandom_inf: list
    Trandom_inf_expanded: list

@dataclass
class ModelContext:
    bnet: str
    model_name: str
    update: str

    primes: dict
    n_nodes: int
    n_sources: int

    percolated_primes: dict
    percolated_bnet: str
    n_perc: int
    n_sources_perc: int
    n_states: int

    stg: object
    scc_dag: object
    attractor_indices: list
    n_attractors: int

    min_trap: list
    n_min_trap: int

    T: np.ndarray
    T_inf: np.ndarray

    sd_indices: list
    Tsd: np.ndarray
    Tsd_inf: np.ndarray
    Tsd_inf_expanded: np.ndarray

    null_indices: list
    Tnull: np.ndarray
    Tnull_inf: np.ndarray
    Tnull_inf_expanded: np.ndarray

    predicted_attractor_indices: list

    random: RandomContext

def _build_model_context(bnet, model_name, update, nsquare_steps, n_random, DEBUG):
    primes = bnet_text2primes(bnet)
    primes = {k: primes[k] for k in sorted(primes)}
    n_nodes = len(primes)
    n_sources = sum(1 for node in primes if primes[node] == [[{node: 0}], [{node: 1}]])
    percolated_primes = percolate(primes, remove_constants=True, copy=True)

    if len(percolated_primes) == 0:
        random = RandomContext(
            n_random=n_random,
            seeds=list(range(n_random)),
            indices=[],
            Trandom=[],
            Trandom_inf=[],
            Trandom_inf_expanded=[],
        )

        return ModelContext(
            bnet=bnet,
            model_name=model_name,
            update=update,

            primes=primes,
            n_nodes=n_nodes,
            n_sources=n_sources,

            percolated_primes=percolated_primes,
            percolated_bnet="",
            n_perc=0,
            n_sources_perc=0,
            n_states=1,

            stg=None,
            scc_dag=None,
            attractor_indices=[],
            n_attractors=1,

            min_trap=[],
            n_min_trap=1,

            T=None,
            T_inf=None,

            sd_indices=[],
            Tsd=None,
            Tsd_inf=None,
            Tsd_inf_expanded=None,

            null_indices=[],
            Tnull=None,
            Tnull_inf=None,
            Tnull_inf_expanded=None,

            predicted_attractor_indices=[],

            random=random,
        )

    percolated_bnet = primes2bnet(percolated_primes)
    stg = primes2stg(percolated_primes, update)
    scc_dag = get_scc_dag(stg)
    attractor_indices = get_attractor_states(scc_dag, as_indices=True, DEBUG=DEBUG)

    n_perc = len(percolated_primes)
    n_sources_perc = sum(1 for node in percolated_primes if percolated_primes[node] == [[{node:0}], [{node:1}]])

    min_trap = compute_trap_spaces(percolated_primes, type_="min")

    T = get_transition_matrix(stg, DEBUG=DEBUG)
    T_inf = nsquare(T, nsquare_steps, DEBUG=DEBUG)

    sd_indices = sd_grouping(percolated_bnet, DEBUG=DEBUG)
    Tsd = compress_matrix(T, sd_indices, DEBUG=DEBUG)
    Tsd_inf = nsquare(Tsd, nsquare_steps, DEBUG=DEBUG)
    Tsd_inf_expanded = expand_matrix(Tsd_inf, sd_indices, DEBUG=DEBUG)

    null_indices = null_grouping(percolated_bnet, DEBUG=DEBUG)
    Tnull = compress_matrix(T, null_indices, DEBUG=DEBUG)
    Tnull_inf = nsquare(Tnull, nsquare_steps, DEBUG=DEBUG)
    Tnull_inf_expanded = expand_matrix(Tnull_inf, null_indices, DEBUG=DEBUG)

    predicted_attractor_indices = get_predicted_attractors(Tsd, sd_indices, as_indices=True, DEBUG=DEBUG)

    random_indices_list = []
    Trandom_list = []
    Trandom_inf_list = []
    Trandom_inf_expanded_list = []
    for i in range(n_random):
        random_indices = random_grouping(sd_indices, null_indices, seed=i, DEBUG=DEBUG)
        Trandom = compress_matrix(T, random_indices, DEBUG=DEBUG)
        Trandom_inf = nsquare(Trandom, nsquare_steps, DEBUG=DEBUG)
        Trandom_inf_expanded = expand_matrix(Trandom_inf, random_indices, DEBUG=DEBUG)

        random_indices_list.append(random_indices)
        Trandom_list.append(Trandom)
        Trandom_inf_list.append(Trandom_inf)
        Trandom_inf_expanded_list.append(Trandom_inf_expanded)

    random = RandomContext(
        n_random=n_random,
        seeds=list(range(n_random)),

        indices=random_indices_list,
        Trandom=Trandom_list,
        Trandom_inf=Trandom_inf_list,
        Trandom_inf_expanded=Trandom_inf_expanded_list
    )

    return ModelContext(
        bnet=bnet,
        model_name=model_name,
        update=update,

        primes=primes,
        n_nodes=n_nodes,
        n_sources=n_sources,

        percolated_primes=percolated_primes,
        percolated_bnet=percolated_bnet,
        n_perc=n_perc,
        n_sources_perc=n_sources_perc,
        n_states=2 ** n_perc,
        
        stg=stg,
        scc_dag=scc_dag,
        attractor_indices=attractor_indices,
        n_attractors=len(attractor_indices),

        min_trap=min_trap,
        n_min_trap=len(min_trap),

        T=T,
        T_inf=T_inf,

        sd_indices=sd_indices,
        Tsd=Tsd,
        Tsd_inf=Tsd_inf,
        Tsd_inf_expanded=Tsd_inf_expanded,

        null_indices=null_indices,
        Tnull=Tnull,
        Tnull_inf=Tnull_inf,
        Tnull_inf_expanded=Tnull_inf_expanded,

        predicted_attractor_indices=predicted_attractor_indices,

        random=random
    )

def get_model_context(
    *,
    bnet,
    model_name,
    update,
    nsquare_steps,
    n_random,
    cache_dir,
    force_rebuild=False,
    DEBUG=False,
):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / f"{model_name}_{update}_{nsquare_steps}_{n_random}.pkl"

    if cache_path.exists() and not force_rebuild:
        with cache_path.open("rb") as f:
            return pickle.load(f)

    ctx = _build_model_context(
        bnet=bnet,
        model_name=model_name,
        update=update,
        nsquare_steps=nsquare_steps,
        n_random=n_random,
        DEBUG=DEBUG,
    )

    with cache_path.open("wb") as f:
        pickle.dump(ctx, f)

    return ctx


def get_model_contexts(
    *,
    model_directory,
    update,
    nsquare_steps,
    n_random,
    cache_dir,
    force_rebuild=False,
    DEBUG=False,
):
    model_directory = Path(model_directory)

    contexts = {}
    model_files = sorted(p for p in model_directory.iterdir() if p.is_file())

    print(f"Preparing contexts for {len(model_files)} models")

    for i, path in enumerate(model_files, start=1):
        print(f"[{i:3d}/{len(model_files)}] {path.name}")

        with path.open() as f:
            bnet = f.read()

        ctx = get_model_context(
            bnet=bnet,
            model_name=path.name,
            update=update,
            nsquare_steps=nsquare_steps,
            n_random=n_random,
            cache_dir=cache_dir,
            force_rebuild=force_rebuild,
            DEBUG=DEBUG,
        )

        contexts[path.name] = ctx

    return contexts
