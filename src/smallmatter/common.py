"""Common utilities with minimum dependencies."""

import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from .pathlib import pathify


class DuplicateError(Exception):
    pass


def lower_uniq(it: Iterable[str]) -> List[str]:
    """Convert strings to lower case without any conflict.

    A typical use-case is to lowercase the column names of a dataframe.

    Sample usage:

    >>> from smallmatter.common import lower_uniq
    >>> lower_uniq(["COL_A", "col_B"])
    ['col_a', 'col_b']

    >>> lower_uniq(["COL_A", "col_B", "col_a", "COL_B"])
    # DuplicateError: Duplicated lowercased members: ['col_a', 'col_b']

    Args:
        it (Iterable[str]): input strings.

    Raises:
        DuplicateError: when duplicate members are present.

    Returns:
        List[str]: lower-cased strings.
    """
    input = list(it)
    lowercased: List[str] = [s.lower() for s in input]
    cnt = Counter(lowercased)

    if len(input) != len(cnt):
        # Friendly error message
        dups = [k for k, v in cnt.items() if v > 1]
        raise DuplicateError(f"Duplicated lowercased members: {dups}")

    return lowercased


DD_T = Dict[str, Any]


class PyExec(object):
    """Helper class to execute valid Python codes and get all the newly created symbols.

    Typical use-case: to implement a data dictionary that can mix-in Python code to construct
    certain variables.
    """

    @staticmethod
    def from_file(path: Union[str, Path, os.PathLike], valid_symbols: Iterable[str] = []) -> DD_T:
        """Execute the Python file and export the symbols as a dictionary."""
        with pathify(path).open("r") as f:
            return PyExec.from_str(f.read(), valid_symbols)

    @staticmethod
    def from_str(s: str, valid_symbols: Optional[Iterable[str]] = None) -> DD_T:
        """Execute the Python codes and export the symbols as a dictionary.

        Args: s (str): Python codes valid_symbols (Optional[Iterable[str]], optional): if None then
            return all symbols, else return only symbols in `valid_symbols`. Defaults to None.

        Returns: DD_T: [description]
        """
        dd_raw: DD_T = {}
        exec(s, dd_raw)
        dd = {k: dd_raw[k] for k in dd_raw}
        if valid_symbols is not None:
            dd = {k: v for k, v in dd.items() if k in set(valid_symbols)}
        return dd
