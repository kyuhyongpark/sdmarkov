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
    model_contexts : iterable of (model_name, ModelContext)
        iterable yielding pre-built ModelContext objects.
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

    for model_name, ctx in model_contexts:
        df = data_function(context=ctx, DEBUG=DEBUG)

        frames.append(df)

        # Crucial: release memory early
        del ctx

    result = pd.concat(frames, ignore_index=True)
    assert not result.empty, "No rows generated"

    if schema is not None:
        for col, dtype in schema.items():
            if col in result.columns:
                result[col] = result[col].astype(dtype)

    return result
