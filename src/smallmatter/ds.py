"""Core data-science utilities."""
import csv
import math
import warnings
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image, ImageChops

from .pathlib import Path2, pathify

# Silence warning due to pandas using deprecated matplotlib API.
# See: https://github.com/pandas-dev/pandas/pull/32444
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)


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


def read_protected_excel(fname: Union[str, Path, PathLike], pwd: str, **kwargs) -> pd.DataFrame:
    """Load a protected Excel file into a pandas.read_excel(..., engine='openpyxl', ...)."""
    import msoffcrypto  # Inner import to avoid imposing dependency to non-users.

    decrypted = BytesIO()
    p = pathify(fname)
    with p.open("rb") as f:
        file = msoffcrypto.OfficeFile(f)
        file.load_key(password=pwd)
        file.decrypt(decrypted)

    if ("engine" in kwargs) and (kwargs["engine"] != "openpyxl"):
        warnings.warn("openpyxl engine is recommended.")

    kwargs["engine"] = "openpyxl"
    return pd.read_excel(decrypted, **kwargs)


def mask_df(df: pd.DataFrame, cols: Sequence[str] = []) -> pd.DataFrame:
    """Mask sensitive columns as "xxx" for display purpose (note: always returns a copy).

    Example:

    >>> import pandas as pd
    >>> from smallmatter.ds import mask_df
    >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
    >>> print(mask_df(df, ['a', 'c']))
        a  b    c
    0  xxx  4  xxx
    1  xxx  5  xxx
    2  xxx  6  xxx

    >>> print(mask_df(df, ['a', 'd']))
    ...
    ValueError: Columns not found in df: {'d'}

    Args:
        df (pd.DataFrame): input dataframe.
        cols (Iterable[str], optional): List of column names to mask. Defaults to [].

    Returns:
        pd.DataFrame: a copy of input dataframe with select columns masked.
    """
    # Do not let df.loc[] silently ignores cols not in df.columns
    invalid_cols = set(cols) - set(df.columns)
    if len(invalid_cols) > 0:
        raise ValueError(f"Columns not found in df: {invalid_cols}")

    df = df.copy()
    fillers = "xxx" if len(cols) == 1 else ["xxx"] * len(cols)
    df.loc[:, cols] = fillers  # This works even when cols == fillers == []
    return df


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


def safe_pystr2json(s) -> str:
    """Convert single-quoted "JSON" string to a double-quoted JSON string.

    Typical use-case: we receive input string `s = "['a', 'b', 'c']"`, which
    were generated essentially by `str(["a", "b", "c"])` and we cannot change
    this part. We can convert the single quotes to double quotes to become a
    valid JSON string. Callers can then deserialized the returned JSON string
    to a Python object using ``json.loads()``.

    This implementation relies on [Black](https://github.com/psf/black) as
    opposed to using `eval()` directly on input string ``s``, to make sure that
    input string contains only JSON payload (i.e., only data, and no code).

    As an example, first we generate synthetic .csv file:

    >>> import json
    >>> import pandas as pd
    >>> df = pd.DataFrame(
        {
            "a": ["apple", "orange"],
            "b": [
                ["red", "green", "blue"],
                ["R", "G", "B"],
            ],
        },
    )
    >>> df.to_csv('/tmp/test-pystr2json.csv', index=False)
    >>> !cat /tmp/test-pystr2json.csv
    a,b
    apple,"['red', 'green', 'blue']"
    orange,"['R', 'G', 'B']"

    Then, we read the .csv file and deserialize the single-quoted string in
    the second column:

    >>> df = pd.read_csv('/tmp/test-pystr2json.csv', low_memory=False)
    >>> df
            a                         b
    0   apple  ['red', 'green', 'blue']
    1  orange           ['R', 'G', 'B']

    >>> # Directly deserialized single-quoted string gives an error.
    >>> df['b'].apply(lambda s: json.loads(s))
    ---------------------------------------------------------------------------
    JSONDecodeError                           Traceback (most recent call last)
    ...
    JSONDecodeError: Expecting value: line 1 column 2 (char 1)

    >>> # Convert python string to JSON string,
    >>> df['b_json'] = df['b'].apply(lambda s: safe_pystr2json(s))
    >>> df
            a                         b                    b_json
    0   apple  ['red', 'green', 'blue']  ["red", "green", "blue"]
    1  orange           ['R', 'G', 'B']           ["R", "G", "B"]

    >>> # then deserialize JSON strings to Python objects
    >>> df['b_obj'] = df['b_json'].apply(lambda s: json.loads(s))
    >>> df[['a', 'b_obj']]
            a               b_obj
    0   apple  [red, green, blue]
    1  orange           [R, G, B]

    >>> type(df.loc[0, 'b_obj'])
    list
    """
    from black import FileMode, format_str

    # Set line_length to have Black reformats the input string to a single line.
    #
    # Otherwise, when reformatted to multiple lines, Black adds trailing commas
    # to the last element in the list or dictionary, which breaks json.loads().
    #
    # Be generous with line_length for pathological cases where input strings
    # contains lots of characters that need to be escaped (e.g., c => \\c)
    return format_str(s, mode=FileMode(line_length=len(s) * 4)).rstrip()


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
            nrows=nrows, ncols=ncols, figsize=(figsize[0] * ncols, figsize[1] * nrows), dpi=dpi, **kwargs
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
# for i in range(1000):
#     title = f"chart-{i:04d}"
#     pd.Series(rand(6)).plot(ax=mp.pop(title), title=title)
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

        Each montage has at most ``page_size`` subplots. This pager automatically saves an existing montage on overflow,
        which occurs when the montage is full and an attempt was made to add a new subplot to it. After the existing
        montage is saved, a new blank montage is created, and the new subplot will be added to it. Callers are expected
        to explicitly save the last montage.

        >>> import pandas as pd
        >>>
        >>> mp = MontagePlotter('output', savefig_kwargs=dict(transparent=TrueFalse))
        >>> for i in range(128):
        >>>     title = f"chart-{i:04d}"
        >>>     pd.Series(rand(6)).plot(ax=mp.pop(title), title=title)
        >>> mp.savefig()  # Save the last montage which may be partially filled.

        >>> mp = MontagePlotter('output', savefig_kwargs=dict(transparent=True))
        >>> ...

        Args:
            prefix (str, optional): Prefix of output filenames. Defaults to "montage".
            page_size (int, optional): Number of subplots per montage. Defaults to 100.
            savefig_kwargs (dict, optional): Keyword arguments to SimpleMatrixPlotter.savefig(), but ``fname`` will be
                overriden by MontagePager.
            kwargs: Keyword arguments to instantiate each montage (i.e., SimpleMatrixPlotter.__init__()).
        """
        self.path = path
        self.montage_path = path / "montages"
        self.individual_path = path / "individuals"
        self.prefix = prefix
        self.page_size = page_size
        self.smp_kwargs = kwargs
        self.smp_kwargs["init_figcount"] = page_size
        self.savefig_kwargs = savefig_kwargs
        self.smp = SimpleMatrixPlotter(**self.smp_kwargs)
        self._i = 0
        self.itemid: List[Any] = []
        self.csvwriter = csv.writer((path / "mappings.csv").open("w"))
        self.csvwriter.writerow(["individual", "title", "montage", "subplot", "row", "col"])

    @property
    def i(self):
        """:int: Sequence number of the current montage (zero-based)."""
        return self._i

    @property
    def filename(self):
        return f"{self.prefix}-{self._i:04d}.png"

    def pop(self, subplot_id: Any = "", **kwargs):
        """Return the next axes, and associate the returned axes with `subplot_id`."""
        if self.smp.i >= self.page_size:
            self.savefig()
            self.smp = SimpleMatrixPlotter(**self.smp_kwargs)
            self._i += 1
        self.itemid.append(subplot_id)
        return self.smp.pop()

    def savefig(self):
        """Save the current montage to a file."""
        # These must be done before smp.savefig() which destroys the underlying figure.
        subplot_cnt = self.smp.i
        bg_rgb = tuple((int(255 * channel) for channel in self.smp.fig.get_facecolor()[:3]))

        if subplot_cnt < 1:
            return

        with BytesIO() as buf:
            # Get the image buffer of the bbox-transformed canvas -- print_figure() in matplotlib/backend_bases.py.
            #
            # NOTE: methods described in https://stackoverflow.com/questions/4325733/save-a-subplot-in-matplotlib) was
            # tested and found not robust. The best outcome was using `exent = ax.get_tightbbox(...)`, which was still
            # not good enought as the next-row title still creeps into individual subplot image (tested with matplotlib
            # 3.2.1 and 3.3.0).
            self.smp.savefig(buf, format="png", **self.savefig_kwargs)
            buf.seek(0)
            im = Image.open(buf)
            im.load()

        im.save(self.montage_path / f"{self.prefix}-{self._i:04d}.png")
        self._save_pieces(im, subplot_cnt, bg_rgb)
        self._save_csv()

    def _save_csv(self):
        """Write row ["individual", "title", "montage-fname", "subplot-idx", "row", "col"]."""
        ncols = self.smp.ncols
        mtg_i = self._i
        mtg_fname = self.filename
        for i, itemid in enumerate(self.itemid):
            row, col = divmod(i, ncols)
            s = str(itemid).encode("unicode-escape").decode("utf-8")
            self.csvwriter.writerow((f"{mtg_i:04d}-{i:02d}.png", s, mtg_fname, i, row, col))
        self.itemid.clear()

    def _save_pieces(
        self,
        im: Image.Image,
        subplot_cnt: int,
        bg_rgb: Tuple[float, float, float] = (255, 255, 255),
        debug: bool = False,
    ):
        """Chop to pieces and save, row-wise."""

        def subplot_size():
            true_nrows = (subplot_cnt // self.smp.nrows) + ((subplot_cnt % self.smp.ncols) != 0)
            true_ncols = min(self.smp.ncols, subplot_cnt)
            h = im.height // true_nrows
            w = im.width // true_ncols
            if debug:
                #    print(f"{im.height=} {im.width=} {subplot_cnt=} {true_nrows=} {true_ncols=} {h=} {w=}")
                print(
                    f"im.height={im.height} im.width={im.width} im.subplot_cnt={subplot_cnt}"
                    f"true_nrows={true_nrows} true_ncols={true_ncols} h={h} w={w}"
                )
            return h, w

        subplot_h, subplot_w = subplot_size()

        def fixed_bbox(i):
            """To crop by fix size, but may produce excess border & plot not centered."""
            row, col = (i // self.smp.nrows), (i % self.smp.ncols)
            up = row * subplot_h
            left = col * subplot_w
            right = left + subplot_w
            bottom = up + subplot_h
            if debug:
                #    print(f"fixed_bbox: {i=} {row=} {col=} {left=} {up=} {right=} {bottom=}")
                print(f"fixed_bbox: i={i} row={row} col={col} left={left} up={up} right={right} bottom={bottom}")
            return left, up, right, bottom

        def tighten(im):
            if self.savefig_kwargs.get("transparent", False):
                bbox = im.getbbox()
            else:
                bg = Image.new("RGB", im.size, bg_rgb)
                diff = ImageChops.difference(im.convert("RGB"), bg)
                bbox = diff.getbbox()

            # Tight crop, but with small pads for a more pleasant view
            cropped = im.crop(pad(*bbox))
            return cropped

        def pad(left: float, up: float, right: float, bottom: float, pixels=4) -> Tuple[float, float, float, float]:
            left = min(left - pixels, 0)
            up = min(up - pixels, 0)
            right = min(right + pixels, im.width)
            bottom = min(bottom + pixels, im.height)
            return (left, up, right, bottom)

        subplot_h, subplot_w = subplot_size()
        for i in range(subplot_cnt):
            cropped_fixed = im.crop(fixed_bbox(i))
            cropped_tight = tighten(cropped_fixed)
            cropped_tight.save(self.individual_path / f"{self._i:04d}-{i:02d}.png")


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
