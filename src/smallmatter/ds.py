"""Core data-science utilities."""
import math
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .pathlib import pathify

# Silence warning due to pandas using deprecated matplotlib API.
# See: https://github.com/pandas-dev/pandas/pull/32444
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

DD_T = Dict[str, Any]


class PyExec(object):
    """Helper class to execute valid Python codes and get all the newly created symbols.

    Typical use-case: to implement a data dictionary that can mix-in Python code to construct certain variables.
    """

    @staticmethod
    def from_file(path: Union[str, Path, os.PathLike], valid_symbols: Iterable[str] = []) -> DD_T:
        """Execute the Python file and export the symbols as a dictionary."""
        with pathify(path).open("r") as f:
            return PyExec.from_str(f.read(), valid_symbols)

    @staticmethod
    def from_str(s, valid_symbols: Iterable[str] = []) -> DD_T:
        """Execute the Python codes and export the symbols as a dictionary."""
        dd_raw: DD_T = {}
        exec(s, dd_raw)
        dd = {k: dd_raw[k] for k in dd_raw}
        return dd


class DFBuilder(object):
    """A helper class to build a dataframe incremental, row-by-row."""

    def __init__(self, dtype=None, columns=None):
        """Construct dataframe row by row.

        Args:
            dtype ([type], optional): see pandas.DataFrame.from_dict. Defaults to None.
            columns ([type], optional): see pandas.DataFrame.from_dict. Defaults to None.
        """
        self.dtype = dtype
        self.columns = columns
        self.rows = []

    def __add__(self, other: Iterable[Any]) -> "DFBuilder":
        """Add a new row to this instance (and modify this instance in-place)."""
        self.rows.append(other)
        return self

    @property
    def df(self):
        """Return the dataframe representation of this instance."""
        return pd.DataFrame.from_dict({i: row for i, row in enumerate(self.rows)}, orient="index", columns=self.columns)


def json_np_converter(o: Union[np.int64, np.float64]) -> Union[int, float]:
    """Convert numpy values to JSON-compliant data type.

    This only works for values. Unfortunately, {json.dump()} still need keys to be strictly Python data types.

    Args:
        o (np.int64 or np.float64): JSON value

    Raises:
        TypeError: if o is not np.int64 or np.float64

    Returns:
        int or float: Python data type that corresponds to o

    Examples:
        >>> import json
        >>> json.dumps(data, default=json_np_converter)
    """
    if isinstance(o, np.int64):
        return int(o)
    if isinstance(o, np.float64):
        return float(o)
    raise TypeError(f"Unknown instance: {type(o)}")


class SimpleMatrixPlotter(object):
    """A simple helper class to fill-in subplot one after another.

    Sample usage using add():

    >>> import pandas as pd
    >>> df = pd.DataFrame({'a': [1,1,1,2,2,2,3,3,3,4,4]})
    >>> gb = df.groupby(by=['a'])
    >>>
    >>> smp = SimpleMatrixPlotter(gb.ngroups)
    >>> for group_name, df_group in gb:
    >>>     ax, _ = smp.add(df_group.plot)
    >>>     assert ax == _
    >>>     ax.set_title(f"Item={group_name}")
    >>> # smp.trim(); plt.show()
    >>> smp.savefig("/tmp/testfigure.png")  # After this, figure & axes are gone.

    Alternative usage using pop():

    >>> smp = SimpleMatrixPlotter(gb.ngroups)
    >>> for group_name, df_group in gb:
    >>>     ax = smp.pop()
    >>>     df_group.plot(ax=ax, title=f"Item={group_name}")
    >>> # smp.trim(); plt.show()
    >>> smp.savefig("/tmp/testfigure.png")  # After this, figure & axes are gone.

    Attributes:
        i (int): Index of the currently free subplot
    """

    def __init__(self, ncols: Optional[int] = None, init_figcount: int = 5, figsize=(6.4, 4.8), dpi=100, **kwargs):
        """Initialize a ``SimpleMatrixPlotter`` instance.

        Args:
            ncols (int, optional): Number of columns. Passing None means to set to sqrt(init_figcount) clipped at
                5 and 20. Defaults to None.
            init_figcount (int, optional): Total number of subplots. Defaults to 5.
            figsize (tuple, optional): size per subplot, see ``figsize`` for matplotlib. Defaults to (6.4, 4.8).
            dpi (int, optional): dot per inch, see ``dpi`` in matplotlib. Defaults to 100.
            kwargs (optional): Keyword argumetns for plt.subplots, but these are ignored and will be overriden:
                ``ncols``, ``nrows``, ``figsize``, ``dpi``.
        """
        # Initialize subplots
        if ncols is None:
            ncols = min(max(5, int(math.sqrt(init_figcount))), 20)
        nrows = init_figcount // ncols + (init_figcount % ncols > 0)

        kwargs = {k: v for k, v in kwargs.items() if k not in {"nrows", "ncols", "figsize", "dpi"}}
        self.fig, _ = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(figsize[0] * ncols, figsize[1] * nrows), dpi=100, **kwargs
        )
        self.axes = self.fig.axes  # Cache list of axes returned by self.fig.axes
        self.fig.subplots_adjust(hspace=0.35)
        self._i = 0  # Index of the current free subplot

        # Warn if initial pixels exceed matplotlib limit.
        pixels = np.ceil(self.fig.get_size_inches() * self.fig.dpi).astype("int")
        if (pixels > 2 ** 16).any():
            warnings.warn(f"Initial figure is {pixels} pixels, and at least one dimension exceeds 65536 pixels.")

    @property
    def i(self):
        """:int: Index of the earliest unused subplot."""
        return self._i

    @property
    def ncols(self):
        """Return the number of columns."""
        return self.axes[0].get_geometry()[1] if len(self.axes) > 0 else 0

    @property
    def nrows(self):
        """Return the number of rows."""
        return self.axes[0].get_geometry()[0] if len(self.axes) > 0 else 0

    @property
    def shape(self):
        """Return a tuple of (rows, cols)."""
        return (self.nrows, self.ncols)

    def add(self, plot_fun, *args, **kwargs) -> Tuple[plt.Axes, Any]:
        """Fill the current free subplot using `plot_fun()`, and set the axes and figure as the current ones.

        Args:
            plot_fun (callable): A function that must accept `ax` keyword argument.

        Returns:
            (plt.Axes, Any): a tuple of (axes, return value of plot_fun).
        """
        ax = self.pop()
        retval = plot_fun(*args, ax=ax, **kwargs)
        return ax, retval

    def pop(self) -> plt.Axes:
        """Get the next axes in this subplot, and set it and its figure as the current axes and figure, respectively.

        Returns:
            plt.Axes: the next axes
        """
        # TODO: extend with new subplots:
        # http://matplotlib.1069221.n5.nabble.com/dynamically-add-subplots-to-figure-td23571.html#a23572
        ax = self.axes[self._i]
        plt.sca(ax)
        plt.figure(self.fig.number)
        self._i += 1
        return ax

    def trim(self):
        """Delete unused subplots."""
        for ax in self.axes[self._i :]:
            self.fig.delaxes(ax)
        self.axes = self.axes[: self._i]

    def savefig(self, *args, **kwargs):
        """Save plotted subplots, then destroy the underlying figure.

        Subsequent operations are undefined and may raise errors.
        """
        self.trim()
        kwargs["bbox_inches"] = "tight"
        self.fig.savefig(*args, **kwargs)
        # Whatever possible ways to release figure
        self.fig.clf()
        plt.close(self.fig)
        del self.fig
        self.fig = None


# NOTE: to simplify the pager implementation, just destroy-old-and-create-new SimpleMatrixPlotter instances.
#       Should it be clear that the overhead of this approach is not acceptable, then reset-and-reuse shall be
#       considered.
#
# As of now, using pager does caps memory usage (in addition to making sure not to hit matplotlib limit of 2^16 pixels
# per figure dimension). The following benchmark to render 10 montages at 100 subplots/montage tops at 392MB RSS, when
# measured on MBP early 2015 model, Mojave 10.14.6, python-3.7.6.
#
# from smallmatter.ds import MontagePager
# import pandas as pd
# mp = MontagePager()
# ser = pd.Series([0,1,2,1,3,2], name='haha')
# for i in range(1000):
#     ser.plot(ax=mp.pop())
# mp.savefig()
class MontagePager(object):
    """A pager to group and save subplots into multiple montage image files."""

    def __init__(
        self,
        path: Path = Path("."),
        prefix: str = "montage",
        page_size: int = 100,
        savefig_kwargs: Dict[str, Any] = {},
        **kwargs,
    ):
        """Render plots to one or more montages.

        Each montage has at most ``page_size`` subplots. This pager automatically saves an existing montage when the
        montage is full and an attempt was made to add a new subplot to it. After the exsting montage is saved, a new
        blank montage is created, and the new subplot will be added to it. Callers are expected to explicitly save the
        last montage.

        Args:
            prefix (str, optional): Prefix of output filenames. Defaults to "montage".
            page_size (int, optional): Number of subplots per montage. Defaults to 100.
            savefig_kwargs (dict, optional): Keyword arguments to SimpleMatrixPlotter.savefig(), but ``fname`` will be
                overriden by MontagePager.
            kwargs: Keyword arguments to instantiate each montage (i.e., SimpleMatrixPlotter.__init__()).
        """
        self.path = path
        self.prefix = prefix
        self.page_size = page_size
        self.smp_kwargs = kwargs
        self.smp_kwargs["init_figcount"] = page_size
        self.savefig_kwargs = savefig_kwargs
        self.smp = SimpleMatrixPlotter(**self.smp_kwargs)
        self._i = 0

    @property
    def i(self):
        """:int: Sequence number of the current montage (zero-based)."""
        return self._i

    def pop(self, **kwargs):
        """Return the next axes."""
        if self.smp.i >= self.page_size:
            self.savefig()
            self.smp = SimpleMatrixPlotter(**self.smp_kwargs)
            self._i += 1
        return self.smp.pop()

    def savefig(self):
        """Save the current montage to a file."""
        # No need to check for empty montage, because Figure.savefig() won't generate output file in such cases.
        self.smp.savefig(self.path / f"{self.prefix}-{self._i:04d}.png", **self.savefig_kwargs)


def plot_binpat(
    img, figsize: Optional[Tuple[float, float]] = None, title: str = "", xlabel: str = "", ylabel: str = ""
):
    """Plot an binary-heatmap image using matplotlib.

    Args:
        img (array-like or PIL image): The image data (see matplotlib.pyplot.imshow())
        figsize (Optional[Tuple[float, float]], optional): Figure size. Defaults to None.
        title (str, optional): Title of the plot. Defaults to "".
        xlabel (str, optional): x-label of the plot. Defaults to "".
        ylabel (str, optional): y-label of the plot. Defaults to "".
    """
    # FIXME: to tidy up.
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.gca().set_yticks([])
    plt.gca().set_xticks([])
    plt.imshow(img, cmap="gray", interpolation="none")
