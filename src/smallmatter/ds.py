"""Core data-science utilities."""
import ast
import csv
import json
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


def pystr2json(s: str) -> str:
    """Convert single-quoted "JSON" string to a double-quoted JSON string.

    Typical use-case: we receive input string `s = "['a', 'b', 'c']"`, which
    were generated essentially by `str(["a", "b", "c"])` and we cannot change
    this part. However, we would still like to convert these strings to valid,
    double-quoted JSON strings for downstreams.

    This implementation relies on ``ast.literal_eval()``. Note that the resulted
    double-quoted string is NOT guaranteed to be a valid JSON string, as clearly
    stated by ``ast.literal_eval()`` documentation where it can yield these
    "Python literal structures: strings, bytes, numbers, tuples, lists, dicts,
    sets, booleans, and None."

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

    Then, we read the .csv file, and show an simple example of downstream task,
    which is to deser the JSON string in the second column to an object:

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

    >>> # Convert python string to JSON string.
    >>> df['b_json'] = df['b'].apply(lambda s: pystr2json(s))
    >>> df
            a                         b                    b_json
    0   apple  ['red', 'green', 'blue']  ["red", "green", "blue"]
    1  orange           ['R', 'G', 'B']           ["R", "G", "B"]

    >>> # Sample downstream task: deserialize JSON strings to Python objects
    >>> df['b_obj'] = df['b_json'].apply(lambda s: json.loads(s))
    >>> df[['a', 'b_obj']]
            a               b_obj
    0   apple  [red, green, blue]
    1  orange           [R, G, B]

    >>> type(df.loc[0, 'b_obj'])
    list
    """
    return json.dumps(ast.literal_eval(s))


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


class CdfResult:
    def __init__(self, cdf_count_name: str, stats_df: pd.DataFrame, cdf: Dict[str, pd.DataFrame]) -> None:
        self.stats_df = stats_df
        self.cdf = cdf
        self.cdf_count_name = self._probe_cdf_count_name()

    @staticmethod
    def renamer_cnt(s: str) -> str:
        return f"{s}_cnt"

    def _probe_cdf_count_name(self) -> str:
        names = {cdf_df.columns[0] for cdf_df in self.cdf.values()}
        assert len(names) == 1
        return names.pop()

    def save_reports(self, output_dir) -> None:
        """Save the reports of a `CdfResult` instance.

        This will save the intermediate stats as an .xlsx file, and each CDF as an .xlsx and an interactive .html.
        """
        from .bkh import BokehPlotter

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Terminologies (I mixed this quite alot, so I feel it warrants a dedicated commentary):
        # - x-axis denote "x_unit / y_basic_unit", e.g., "date_cnt / item".
        #   * x_unit comes from cdf_df.
        #   * y_basic_unit comes from stats_df.
        # - y-axis is the probability, and the raw count is y_unit, e.g., "item_cnt".
        #   * y_unit comes from cdf_df.
        #
        # Grand example: x_unit="date_cnt", y_basic_unit="item", y_unit="item_cnt".
        #
        # When in doubt, see the sample dataframes in the docstring of cdf().
        y_basic_unit = "#".join(self.stats_df.index.names)
        y_unit = self.cdf_count_name

        # Save the intermediate statistics
        stats_fname = output_dir / f"stats-by-{y_basic_unit}.xlsx"
        self.stats_df.to_excel(stats_fname, index=True, freeze_panes=(1, 1))

        # Save the cdf statistics.

        for col, cdf_df in self.cdf.items():
            x_unit = cdf_df.index.name

            # Save as table
            self.cdf[col].to_excel(
                output_dir / f"cdf-{x_unit}-per-{y_basic_unit}.xlsx", index=True, freeze_panes=(1, 0)
            )

            # Save as interactive html
            bp = BokehPlotter(
                self.cdf[col].reset_index().rename({col: "x", "cdf": "y"}, axis=1),
                plot_width=960,
                plot_height=480,
                title=f"CDF of {x_unit} / {y_basic_unit}",
                x_label=f"{x_unit} / {y_basic_unit}",
                y_label="Cumulative Probability",
                hover_tooltips={
                    f"cumprob": "@y",
                    f"{y_unit}": f"@{{{y_unit}}}",
                    f"{x_unit}": "@x",
                    "Here to left (aka cum-sum)": "@cum_sum",
                    "After here (aka right-hand side)": "@rhs",
                },
            )
            bp.gen_plot()
            # bokeh.plotting.show(bp.plot)
            bp.save_html(output_dir / f"cdf-{x_unit}-per-{y_unit}.html")


def cdf(df, cdf_count_name="count", rename=CdfResult.renamer_cnt, **kwargs) -> CdfResult:
    """For each group, compute the cdf of each column.

    This is a convenience wrapper to `stats_by()` followed by a number of `get_cdf()` calls, and renaming of columns.
    Use `rename` to customize the column names of the intermediate statistics, and `cdf_count_name` to customize the
    `count` column name in the CDF dataframe.

    Usage:

    >>> items = ['ab', 'cd', 'ef', 'ef', 'ef']
    >>> dates = ['2019-06-07', '2017-08-19', '2018-04-24', '2019-01-16', '2019-01-16']
    >>> channels = ['c1', 'c2', 'c1', 'c2', 'c3']
    >>> tags = ['t', 't', 't', 't', 't']
    >>> df = pd.DataFrame({'item': items, 'date': dates, 'channel': channels, 'tag': tags})
    >>> df
      item        date channel tag
    0   ab  2019-06-07      c1   t
    1   cd  2017-08-19      c2   t
    2   ef  2018-04-24      c1   t
    3   ef  2019-01-16      c2   t
    4   ef  2019-01-16      c3   t

    >>> cdf_results = cdf(df, by='item', cdf_count_name='item_cnt', agg_funcs={'date': 'nunique', 'channel': 'nunique'})
    >>> cdf_results.stats_df
          date_cnt  channel_cnt
    item
    ab           1            1
    cd           1            1
    ef           2            3

    >>> list(cdf_results.cdf)
    ['date_cnt', 'channel_cnt']

    >>> cdf_results.cdf['date']
              item_cnt       cdf  cum_sum  rhs
    date_cnt
    1                2  0.666667        2    1
    2                1  1.000000        3    0

    >>> cdf_results.cdf['channel']
                 item_cnt       cdf  cum_sum  rhs
    channel_cnt
    1                   2  0.666667        2    1
    3                   1  1.000000        3    0

    Notice how only columns in `agg_funcs` are in the results.

    Args:
        df (pd.DataFrame): [description]
        by ([type]): as in pandas.DataFrame.groupby()
        rename ([type]): as in pandas.DataFrame()
        kwargs ([type]): see `stats_by()`

    Returns:
        Dict[str, pd.DataFrame]: {'colname': cdf_dataframe returned by get_cdf()}
    """
    stats_df = stats_by(df, rename=rename, **kwargs)
    cdf_results: Dict[str, pd.DataFrame] = {}
    for col in stats_df.columns:
        raw_cnt: pd.Series = stats_df[col].reset_index().groupby(by=col).count().iloc[:, 0]
        cdf_results[col] = get_cdf(raw_cnt)
        if cdf_count_name != "count":
            cdf_results[col] = cdf_results[col].rename({"count": cdf_count_name}, axis=1)

    return CdfResult(cdf_count_name, stats_df, cdf_results)


################################################################################
# Low-level functions to support cdf functions
################################################################################
def stats_by(df: pd.DataFrame, by, agg_funcs, rename={}) -> pd.DataFrame:
    """For each group, compute the aggregate statistics for columns.

    >>> items = ['ab', 'cd', 'ef', 'ef', 'ef']
    >>> dates = ['2019-06-07', '2017-08-19', '2018-04-24', '2019-01-16', '2019-01-16']
    >>> channels = ['c1', 'c2', 'c1', 'c2', 'c3']
    >>> df = pd.DataFrame({'item': items, 'date': dates})
    >>> df
      item        date channel
    0   ab  2019-06-07      c1
    1   cd  2017-08-19      c2
    2   ef  2018-04-24      c1
    3   ef  2019-01-16      c2
    4   ef  2019-01-16      c3

    >>> stats_by(df, by='item', agg_funcs={'date': 'nunique'}, rename={'date': 'day_cnt'})
          day_cnt
    item
    ab          1
    cd          1
    ef          2

    Notice how only columns in `agg_funcs` are in the results.

    Args:
        df (pd.DataFrame): [description]
        by ([type]): as in see pandas.DataFrame.groupby()
        agg_funcs ([type]): as in pandas.DataFrame.agg()
        rename ([type]): as in pandas.DataFrame.rename(mapper=...)

    Returns:
        pd.DataFrame: [description]
    """
    return df.groupby(by=by).agg(agg_funcs).rename(mapper=rename, axis=1)


def get_cdf(raw_cnt: pd.Series) -> pd.DataFrame:
    """Compute CDF and related metrics from a Pandas Series.

    Typically used in conjuction with `stats_df`.

    >>> stats_df = stats_by(df, by='item', agg_funcs={'date': 'nunique'}, rename={'date': 'day_cnt'})
    >>> stats_df
          day_cnt
    item
    ab          1
    cd          1
    ef          2

    >>> raw_cnt_df = stats_df['day_cnt'].reset_index().groupby(by='day_cnt').count()
    >>> raw_cnt_df
             item
    day_cnt
    1           2
    2           1

    >>> cdf_df = get_cdf(raw_cnt_df.iloc[:, 0])
    >>> cdf_df
             count       cdf  cum_sum  rhs
    day_cnt
    1            2  0.666667        2    1
    2            1  1.000000        3    0
    """
    # CDF
    cum_sum = raw_cnt.cumsum()
    cdf = cum_sum / cum_sum.iloc[-1]

    # Right-hand side
    rhs = cum_sum.iloc[-1] - cum_sum
    return pd.DataFrame({"count": raw_cnt, "cdf": cdf, "cum_sum": cum_sum, "rhs": rhs})
