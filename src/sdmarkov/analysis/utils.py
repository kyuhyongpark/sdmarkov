import os

import numpy as np
import pandas as pd

def load_or_generate_data(
    data_csv,
    generate_data=False,
    data_function=None,
    model_directory=None,
    update="asynchronous",
    n_random=30,
    debug=False,
):
    """
    Load cached network-level results or generate them from model files.

    Parameters
    ----------
    data_csv : str
        Path to cached CSV.
    model_directory : str
        Directory with model files.
    data_function : callable
        Function to generate rows from a single model.
    update : str
        Update scheme passed to get_data.
    n_random : int
        Number of random runs per model.
    debug : bool
        Verbosity/debug flag.
    generate_data : bool
        Force generation even if CSV exists.

    Returns
    -------
    pd.DataFrame
        Combined results from all models.
    """

    if not generate_data and os.path.exists(data_csv):
        df = pd.read_csv(data_csv)
        print(f"Loaded cached results from {data_csv}.")
        return df

    all_rows = []

    model_files = sorted(
        f for f in os.listdir(model_directory)
        if os.path.isfile(os.path.join(model_directory, f))
    )

    n_files = len(model_files)
    print(f"Generating data from {n_files} models...")

    for i, filename in enumerate(model_files, start=1):
        path = os.path.join(model_directory, filename)
        print(f"[{i:3d}/{n_files}] Analyzing: {filename}")

        with open(path) as f:
            content = f.read()

        rows_df = data_function(
            bnet=content,
            bnet_name=filename,
            update=update,
            num_runs=n_random,
            DEBUG=debug,
        )

        all_rows.append(rows_df)

    df = pd.concat(all_rows, ignore_index=True)
    assert not df.empty, "No rows generated"

    df["update_scheme"] = update
    df.to_csv(data_csv, index=False)
    print(f"Saved results to {data_csv}")

    return df


def add_classification_metrics(df):
    """
    Adds precision, recall, specificity, and NPV columns.
    Safe against division-by-zero.
    """

    df = df.copy()

    df["precision"] = np.where(
        (df.TP + df.FP) > 0,
        df.TP / (df.TP + df.FP),
        np.nan,
    )

    df["recall"] = np.where(
        (df.TP + df.FN) > 0,
        df.TP / (df.TP + df.FN),
        np.nan,
    )

    df["specificity"] = np.where(
        (df.TN + df.FP) > 0,
        df.TN / (df.TN + df.FP),
        np.nan,
    )

    df["npv"] = np.where(
        (df.TN + df.FN) > 0,
        df.TN / (df.TN + df.FN),
        np.nan,
    )

    return df


def network_level_metrics_df(df, random_method="random_mc", agg="mean"):
    """
    Aggregate numeric columns for the random method per network,
    leave deterministic methods unchanged, preserve non-numeric metadata,
    drop 'run', and sort by bnet then method.
    """
    # deterministic methods (leave as-is)
    det = df[df.method != random_method]

    # random method
    rnd = df[df.method == random_method]

    # identify numeric and non-numeric columns
    numeric_cols = rnd.select_dtypes(include="number").columns.tolist()
    non_numeric_cols = rnd.select_dtypes(exclude="number").columns.difference(["bnet"]).tolist()

    # aggregate numeric columns per network
    rnd_numeric_agg = rnd.groupby("bnet", as_index=False)[numeric_cols].agg(agg)

    # preserve non-numeric columns (take first value per network)
    rnd_non_numeric = rnd.groupby("bnet", as_index=False)[non_numeric_cols].first()

    # merge numeric and non-numeric results
    rnd_agg = pd.merge(rnd_numeric_agg, rnd_non_numeric, on="bnet")

    # combine deterministic and aggregated random
    combined = pd.concat([det, rnd_agg], ignore_index=True)

    # drop 'run' column if it exists
    if "run" in combined.columns:
        combined = combined.drop(columns=["run"])

    # sort by bnet first, then method
    combined = combined.sort_values(by=["bnet", "method"]).reset_index(drop=True)

    return combined
