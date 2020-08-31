# Simplified from https://github.com/verdimrc/pdutil/blob/4a1d9b3ed3a09aa9df01c1ecfb97f976f9fd3b72/pdcheck.py

"""Checks dataframe, either treating all values as strings or as auto-detected dtypes.

Note that python-3.6+ permits '_' as floating point separator, thus this module will treat '_'-separated float string
as valid floating points.
"""

import json
import re
from typing import Any, List, Optional

import pandas as pd

ALPHANUMERIC_REGEXP = re.compile(r"[0-9a-zA-Z]")
FENCE_REGEXP = re.compile(r"^[^0-9a-zA-Z]|[^0-9a-zA-Z\.]$")


def read_csv(fname: str, as_str=False, **kwargs) -> pd.DataFrame:
    """Wrapper to pd.read_csv() with two presets: load csv file as string, or auto-detect dtype as per pandas default.

    :param as_str: load all columns as strings (implies na_filter=False)."""

    # Column data types
    dtype: Optional[type]
    if as_str:
        dtype = str
        na_filter = False
    else:
        dtype = None
        na_filter = True

    # Load .csv into a dataframe, with proper encoding and selected columns.
    df = pd.read_csv(fname, low_memory=False, dtype=dtype, na_filter=na_filter, **kwargs)
    return df


def check_dtypes_distribution(df, max_value_to_show=10):
    """Get dtypes distributions of dataframe.

    Equivalent to check_columns() and check_datapoints_dtype().
    """
    dfs = [check_columns(df, max_value_to_show), check_datapoints_dtype(df).iloc[:, 1:]]
    return pd.concat(dfs, axis=1)


def extract_str_values(df, filter=None) -> pd.DataFrame:
    """Get string values from each column in dataframe, optionally only those filtered by function `filter`."""
    column: List[str] = []
    str_values: List[str] = []
    d = {"column": column, "str_values": str_values}

    for i in df.columns:
        ser = df[i].drop_duplicates()
        column.append(i)
        if filter:
            str_filter = ser.apply(lambda x: not is_number_str(x) and filter(x))
        else:
            str_filter = ser.apply(lambda x: not is_number_str(x))
        str_ser = ser[str_filter]
        str_values.append(json.dumps(str_ser.tolist(), ensure_ascii=False))

    return pd.DataFrame(d, columns=["column", "str_values"])


def check_possible_dtype(df):
    """Guess dtypes for each column in a dataframe, where dataframe must contains only string values.

    Raise an exception if dataframe contains non-string values.

    :param df: a DataFrame whose all values must be strings.
    """
    column = []
    int_cnt = []
    dec_cnt = []
    str_cnt = []
    d = {"column": column, "int_cnt": int_cnt, "dec_cnt": dec_cnt, "str_cnt": str_cnt}

    for i in df.columns:
        ser = df[i].drop_duplicates()
        column.append(i)
        int_cnt.append(ser.apply(lambda x: is_int_str(x)).sum())
        dec_cnt.append(ser.apply(lambda x: is_dec_str(x)).sum())
        str_cnt.append(ser.apply(lambda x: not is_number_str(x)).sum())

    dtype_options_df = pd.DataFrame(d, columns=["column", "int_cnt", "dec_cnt", "str_cnt"])

    # Best-effort guess on dtype
    guessed_dtype = dtype_options_df.apply(guess_dtype, axis=1).rename("guessed_type_for_non_nan")

    return pd.concat([dtype_options_df, guessed_dtype], axis=1)


def is_nonalphanum_str(s) -> bool:
    """Check whether string `s` is '', or contains no alphanumeric character at all."""
    try:
        _ = s.encode("ascii")
        return True if s == "" else not ALPHANUMERIC_REGEXP.search(s)
    except UnicodeEncodeError:
        # Treat string with non-CJK_space characters as "not suspicious".
        # NOTE:
        # - \u3000 is CJK whitespace
        # - There're other unicode whitespaces listed here: https://stackoverflow.com/a/37903645
        return b"\\u3000" in s.encode("unicode_escape")
    except:
        print(s)
        raise


def is_fenced_str(s: str, ignore_number: bool = True) -> bool:
    """Check whether string `s` starts or ends with a non-alphanumeric character."""
    try:
        _ = s.encode("ascii")
        is_fenced = bool(FENCE_REGEXP.search(s))
        if ignore_number and is_number_str(s):
            return False
        return is_fenced

    except UnicodeEncodeError:
        # Treat string with non-CJK_space characters as "not suspicious".
        # NOTE:
        # - \u3000 is CJK whitespace
        # - There're other unicode whitespaces listed here: https://stackoverflow.com/a/37903645
        return b"\\u3000" in s.encode("unicode_escape")
    except:
        print(s)
        raise


################################################################################
# L2 functions
################################################################################
def check_columns(df, max_item_to_show=10):
    """Column dtype are computed from non-NaN values to prevent int64 columns becomes float64."""
    column = []
    dtyp = []
    uniq_cnt = []
    data_cnt = []
    nan_cnt = []
    sample_value = []
    d = {
        "column": column,
        "dtype": dtyp,
        "uniq_cnt": uniq_cnt,
        "data_cnt": data_cnt,
        "nan_cnt": nan_cnt,
        "sample_value": sample_value,
    }

    for i in df.columns:
        col = df[i]
        uniques = col.unique()
        cnt = len(col)

        column.append(i)
        dtyp.append(col.dropna().dtype)
        uniq_cnt.append(len(uniques))
        nan_cnt.append(cnt - col.count())
        data_cnt.append(cnt)

        # Convert to string, otherwise jupyter notebook display without padding spaces
        # sample_value.append(str(uniques[:max_item_to_show].tolist()))
        sample_value.append(json.dumps(uniques[:max_item_to_show].tolist()))

    return pd.DataFrame(d, columns=["column", "dtype", "uniq_cnt", "data_cnt", "nan_cnt", "sample_value"])


def check_datapoints_dtype(df):
    """Only dtypes of non-NaN values to prevent int64 columns become float64."""
    column = list(df.columns)
    dtypes = []
    dtype_cnt = []
    d = {"column": column, "dtypes": dtypes, "dtype_cnt": dtype_cnt}

    for i in column:
        dt = df[i].dropna().apply(lambda x: x.__class__.__name__).unique().tolist()
        dtypes.append(json.dumps(dt))
        dtype_cnt.append(len(dt))

    return pd.DataFrame(d, columns=["column", "dtypes", "dtype_cnt"])


################################################################################
# L3 functions
################################################################################
def guess_dtype(x):
    if x["str_cnt"] > 0:
        return "str"
    if x["dec_cnt"] > 0:
        return "float"
    if x["int_cnt"] > 0:
        return "int"
    return "UNKNOWN"


def is_int_str(x: str):
    return x.isnumeric()


def is_dec_str(x: str, strict=True):
    try:
        float(x)
    except ValueError:
        return False
    else:
        if strict:
            return ("." in x) or ("e" in x.lower())
        else:
            return True


def is_number_str(x: Any):
    return is_dec_str(x, strict=False)
