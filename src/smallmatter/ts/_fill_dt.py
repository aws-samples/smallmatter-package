"""This module is based on the repository ``aws-samples/amazon-sagemaker-gluonts-entrypoint``,
file `src/gluonts_nb_utils/__init__.py`.

The repo itself is part of an AWS blog post on
[Demand Forecasting using Amazon SageMaker and GluonTS at Novartis AG (Part 4/4)](https://aws.amazon.com/blogs/industries/novartis-ag-uses-amazon-sagemaker-and-gluonts-for-demand-forecasting/).
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd


def fill_dt_all(df: pd.DataFrame, ts_id: List[str], **kwargs) -> pd.DataFrame:
    """Convert fragmented (i.e., sparse) timeseries to contiguous (i.e., dense) timeseries.

    Dataframe `df` consists of multiple timeseries, where each timeseries is a set of 1+ rows
    with the same `ts_id` values.

    Dataframe `df` must have the timestamp column named as ``x`` and its type must be
    `pd.Timestamp`.

    Essentially, this function fills the head, internal fragmentations, and tail of timeseries.

    Example:

    >>> # First, create a sample input dataframe of two sparse timeseries.
    >>> import pandas as pd
    >>> from smallmatter.ts import fill_dt_all
    >>>
    >>> df = pd.DataFrame({
    >>>         'sku': ['item-a', 'item-a', 'item-a', 'item-b', 'item-b'],
    >>>         'loc': ['city-01', 'city-01', 'city-01', 'city-02', 'city-02'],
    >>>         'x': ['2021-06-06', '2021-06-09', '2021-06-11', '2021-06-08', '2021-06-12'],
    >>>         'y': [1, 2, 3, 4, 5],
    >>>     },
    >>> )
    >>> # Important: make sure column x is timestamp, and not string! This example uses daily
    >>> # frequency, but other frequencies are also acceptable.
    >>> df['x'] = pd.to_datetime(df['x'], format='%Y-%m-%d')
    >>> df
          sku      loc          x  y
    0  item-a  city-01 2021-06-06  1
    1  item-a  city-01 2021-06-09  2
    2  item-a  city-01 2021-06-11  3
    3  item-b  city-02 2021-06-08  4
    4  item-b  city-02 2021-06-12  5

    >>> # The timestamps in the input dataframe can be out-of-order. Let's demonstrate this by
    >>> # intentionally shuffle the dataframe.
    >>> df = df.sample(frac=1).reset_index(drop=True)
    >>> df
          sku      loc          x  y
    0  item-b  city-02 2021-06-08  4
    1  item-a  city-01 2021-06-11  3
    2  item-a  city-01 2021-06-09  2
    3  item-b  city-02 2021-06-12  5
    4  item-a  city-01 2021-06-06  1

    >>> # It's time to fill-in the "holes" aka internal fragmentations, to convert each
    >>> # sparse timeseries to dense ones. Note that `dates` is an argument of
    >>> # `smallmatter.ts.fill_dt`, so please refer to its docstring.
    >>> df_dense = fill_dt_all(df, ts_id=['sku', 'loc'], dates=('min', 'max', 'D'))
               x sku      loc    y
    0 2021-06-06   a  city-01  1.0
    1 2021-06-07   a  city-01  0.0
    2 2021-06-08   a  city-01  0.0
    3 2021-06-09   a  city-01  2.0
    4 2021-06-10   a  city-01  0.0
    5 2021-06-11   a  city-01  3.0
    0 2021-06-08   b  city-02  4.0
    1 2021-06-09   b  city-02  0.0
    2 2021-06-10   b  city-02  0.0
    3 2021-06-11   b  city-02  0.0
    4 2021-06-12   b  city-02  5.0

    >>> # Optionally, reset the index of the output dataframe.
    >>> df_dense.reset_index(drop=True, inplace=True)
    >>> df_dense


    Args:
        df (pd.DataFrame): a long dataframe where rows with the same ``ts_id`` represent a single
            timeseries.
        ts_id (list, optional): a key or composite key of timeseries.
        kwargs: additional customization to :func:`~.fill_dt`. Please refer to that function's
            docstring for available customizations.

    Returns:
        pd.DataFrame: a long dataframe consisting contiguous timeseries.
    """
    ts = df.groupby(ts_id, as_index=False, group_keys=False).apply(fill_dt, **kwargs)
    return ts


def fill_dt(
    df: pd.DataFrame,
    dates: Union[pd.DatetimeIndex, Tuple[str, str, str]],
    freq: str = "D",
    fillna_kwargs: Optional[Dict[str, Any]] = None,
    resample: str = "sum",
    resample_kwargs: Dict[str, Any] = {},
) -> pd.DataFrame:
    """Make sure each timeseries has contiguous days, then optionally downsampled.

    Notes the semantic of this function:
    - the input dataframe `df` must have either column "x", or indexed with "x" only.
    - number columns undergo the `pd.DataFrame.fillna` procedure.
    - a non-number column is assumed to be a categorical feature of the whole timeseries (and NOT of
      a timestamp). In other words, each non-number column equals to what gluonts calls as
      "static category feature": given a timeseries which is 1+ rows, the value of this column on
      all the rows are the same. If the input ``df`` mistakenly assigns multiple values to this
      column, then this function will use the value of the first input row in the output dataframe.

    Arguments:
        df (pd.DataFrame): an input dataframe consisting of only one sparse timeseries.
        dates (pd.DatetimeIndex or Tuple[str, str, str]): new timestamp index.
            - If `pd.DatetimeIndex`, then this is typically created by
              ``pd.date_range("yyyy-mm-dd", "yyyy-mm-dd", freq="D"))``.
            - If `Tuple[str, str, str]`, then:
              - ``dates[0]`` will be either string ``yyyy-mm-dd`` or string ``min``,
              - ``dates[1]`` will be either ``yyyy-mm-dd`` or string ``max``, and
              - ``dates[2]`` will be the frequency of the original index.
        freq (str): the output frequency. After `df` is reindexed, further downsample to this freq.
        fillna_kwargs (Dict[str, Any], optional):  Use None for demand, ``{ 'method': 'ffill'}`` for
            price. Defaults to None.
        resample_fn (str, optional): Set to ``sum`` for demand, ``max`` for price curves. Defaults
            to ``sum``.
        resample_kwargs (Dict[str, Any], optional): [description]. Defaults to ``{}``.

    Returns:
        pd.DataFrame: a dataframe of a dense timeseries, where each row denotes a timestamp in
            the timeseries, and the whole timeseries has contiguous timestamps according to the
            requested frequency `freq`. If the input dataframe `df` is indexed by ``x``, then the
            this ``x`` becomes a regular column in the output dataframe.
    """
    X = "x"
    if X in df.columns:
        df = df.set_index(X).copy()

    if not isinstance(dates, pd.DatetimeIndex):
        # Must be Tuple[str, str, str]
        start, end, freq_ori = dates
        if start == "min":
            start = df.index.min()
        if end == "max":
            end = df.index.max()
        dates = pd.date_range(start, end, freq=freq_ori)

    # Pre-compute nan-filler.
    # - number columns: fillna with 0.0
    # - non-number columns: fillna with the 1st non-NA
    nan_repl = df.iloc[0:1, :].reset_index(drop=True)
    for i in range(nan_repl.shape[1]):
        if pd.api.types.is_numeric_dtype(type(nan_repl.iloc[0, i])):
            nan_repl.iloc[0, i] = 0.0
    nan_repl = {k: v[0] for k, v in nan_repl.to_dict().items()}

    # Re-index timeseries to contiguous days
    if fillna_kwargs is None:
        daily_binpat = df.reindex(dates).fillna(value=nan_repl)
    else:
        daily_binpat = df.reindex(dates).fillna(**fillna_kwargs)
        # For non-number columns, always use the value from the first row
        col_to_refill = {k: v for k, v in nan_repl.items() if not pd.api.types.is_numeric_dtype(type(v))}
        for k, v in col_to_refill.items():
            daily_binpat[k] = v
    daily_binpat.index.name = df.index.name

    # FIXME: should be: if output_freq == input_freq
    if freq == "D":
        return daily_binpat.reset_index()

    # Downsample y if necessary.
    downsampled_binpat = daily_binpat.resample(freq)
    resample_fn = getattr(downsampled_binpat, resample)
    downsampled_binpat = resample_fn(**resample_kwargs)

    # Resample will drop non-number columns, so we need to restore them.
    col_to_reinsert = {k: v for k, v in nan_repl.items() if not pd.api.types.is_numeric_dtype(type(v))}
    for k, v in col_to_reinsert.items():
        downsampled_binpat[k] = v

    return downsampled_binpat.reset_index()
