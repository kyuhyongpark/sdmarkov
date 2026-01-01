import os

import numpy as np
import pandas as pd

def generate_dataframe(
    model_contexts,
    data_function,
    *,
    schema=None,
    DEBUG=False,
):
    """
    Generate a combined pandas DataFrame from multiple ModelContext objects.

    Parameters
    ----------
    model_contexts : dict[str, ModelContext]
        Dictionary mapping model names to pre-built ModelContext objects.
    data_function : callable
        Function that takes a single ModelContext and returns a pandas DataFrame.
    schema : dict[str, str] or None, optional
        Optional mapping of column names to pandas dtypes.
        Only columns present in the final DataFrame are cast.
    DEBUG : bool, optional
        Passed through to data_function.

    Returns
    -------
    pd.DataFrame
        Concatenated results from all model contexts.
    """

    frames = []

    for model_name, ctx in model_contexts.items():
        df = data_function(context=ctx, DEBUG=DEBUG)

        frames.append(df)

    result = pd.concat(frames, ignore_index=True)
    assert not result.empty, "No rows generated"

    if schema is not None:
        for col, dtype in schema.items():
            if col in result.columns:
                result[col] = result[col].astype(dtype)

    return result



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
    drop 'run', and sort by model then method.
    """
    # deterministic methods (leave as-is)
    det = df[df.method != random_method]

    # random method
    rnd = df[df.method == random_method]

    # identify numeric and non-numeric columns
    numeric_cols = rnd.select_dtypes(include="number").columns.tolist()
    non_numeric_cols = rnd.select_dtypes(exclude="number").columns.difference(["model"]).tolist()

    # aggregate numeric columns per network
    rnd_numeric_agg = rnd.groupby("model", as_index=False)[numeric_cols].agg(agg)

    # preserve non-numeric columns (take first value per network)
    rnd_non_numeric = rnd.groupby("model", as_index=False)[non_numeric_cols].first()

    # merge numeric and non-numeric results
    rnd_agg = pd.merge(rnd_numeric_agg, rnd_non_numeric, on="model")

    # combine deterministic and aggregated random
    combined = pd.concat([det, rnd_agg], ignore_index=True)

    # drop 'run' column if it exists
    if "run" in combined.columns:
        combined = combined.drop(columns=["run"])

    # sort by model first, then method
    combined = combined.sort_values(by=["model", "method"]).reset_index(drop=True)

    return combined
