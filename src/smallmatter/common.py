"""Common utilities with minimum dependencies."""

from collections import Counter
from typing import Any, Iterable, List


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
