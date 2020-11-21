"""Common utilities with minimum dependencies."""

import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from .pathlib import pathify

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


def lowerable(src: Sequence[str], dst: Sequence[str]) -> bool:
    """Check whether all strings in `src` can be lower_uniq()-ed to `dst` in the exact same sequence.

    Typical use-case: to rename column names of multiple dataframes to same lower-case forms.

    Sample usage:

    >>> from smallmatter.common import lowerable
    >>> lowerable(src=['col_A', 'COL_B'], dst=['col_a', 'col_b'])
    # True

    >>> lowerable(src=['COL_B', 'col_A'], dst=['col_a', 'col_b'])
    # False

    >>> lowerable(src=['col_A', 'COL_B', 'Col_c'], dst=['col_a', 'col_b'])
    # False

    >>> lowerable(src=[], dst=['col_a', 'col_a'])
    # DuplicateError: Duplicated dst: ['col_a']

    >>> lowerable(src=['col_a', 'col_a'], dst=[])
    # False

    >>> lowerable(src=['col_b', 'COL_B'], dst=[])
    # False

    Args:
        src (Iterable[str]): source strings in mixed cases
        dst (Iterable[str]): target lowercase strings; must be unique.

    Raises:
        DuplicateError: when `dst` contains duplicated members.


    Returns:
        List[str]: lowerable source strings
    """
    # Sanity check on outputs
    if len(dst) != len(set(dst)):
        # Friendly error message
        cnt = Counter(dst)
        dups = [k for k, v in cnt.items() if v > 1]
        raise DuplicateError(f"Duplicated dst: {dups}")

    try:
        lowered_src = lower_uniq(src)
        return lowered_src == dst
    except DuplicateError:
        return False
