import numpy as np
import pandas as pd


# Row-wise metric derivation
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

# Aggregation / reduction
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
