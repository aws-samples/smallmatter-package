import argparse
import csv
import json
import pprint as pp
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union

import matplotlib.pyplot as plt
import pandas as pd

from ..ds import MontagePager
from ..pathlib import Path2, S3Path
from . import fill_dt

try:
    # Workaround to regression in matplotlib-3.3.0: wrong dates plotted on x-axis.
    # https://github.com/matplotlib/matplotlib/issues/18010#issuecomment-663738841
    plt.rcParams["date.epoch"] = "0000-12-31T00:00"
except KeyError:
    pass

TsidDeser = Union[str, List[str]]
Tsid = List[str]


def get_tsid(s: str) -> Tsid:
    tsid_deser: TsidDeser = json.loads(s)
    tsid_cols: Tsid = [tsid_deser] if isinstance(tsid_deser, str) else tsid_deser
    return tsid_cols


def _prep_output_dirs(prefix: Path):
    montage_dir, individual_dir = (prefix / _ for _ in ("montages", "individuals"))
    for d in (montage_dir, individual_dir):
        d.mkdir(parents=True, exist_ok=True)
    return montage_dir, individual_dir


def main(
    output_dir: Path2,
    input: Path2,
    kind: str = "event",
    x: str = "x",
    y: str = "y",
    tsid_json: str = '"tsid"',
    start: str = "2017-01-01",
    end: str = datetime.today().strftime("%Y-%m-%d"),
    freq: str = "W",
    transparent: bool = False,
    create_output_dir: bool = True,
) -> None:
    """Plot all timeseries.

    Usage::

        python -m smallmatter.ts.plot_all \
            test/refdata/plot_all_ts.csv \
            -x replen_date \
            -y qty \
            --tsid '["sku", "location"]'

    Args:
        output_dir (Path2): Output directory.
        input (Path2): Filename of input .csv.
        x (str, optional): Column name for timestamp (i.e., x-axis). Defaults to "x".
        y (str, optional): Column name for y-axis. Defaults to "y".
        tsid_json (str, optional): Columns for timeseries identifier in JSON format. Defaults to '"tsid"'.
        start (str, optional): Start date in YYYY-MM-DD format. Defaults to "2017-01-01".
        end (str, optional): End date in YYYY-MM-DD format. Defaults to today.
        freq (str, optional): Frequency used in the plots. Defaults to "W".
        transparent (bool, optional): Whether to use transparent background. Defaults to False.
        create_output_dir (bool, optional): Whether to output directory if missing. Defaults to True.
    """
    # This must preceed any other stuffs (unless we want to venture to the land of inspect module)
    metadata = locals()

    # Prep. output dirs
    if create_output_dir and not isinstance(output_dir, S3Path):
        output_dir.mkdir(parents=True, exist_ok=True)
    montage_dir, individual_dir = _prep_output_dirs(output_dir)

    # Flush the metadata out
    with (output_dir / "tslen.metadata").open("w") as f:
        f.write(
            """################################################################################
# This is not JSON format, nor a valid Python file.
#
# Rather, this is simply the __repr__() of a Python dictionary, and is mainly
# intended for human inspection.
#
# In future, we may decide to serialize the dictionary for downstream tasks.
################################################################################

"""
        )

        pp.pprint(metadata, stream=f)

    # Read input .csv
    tsid = get_tsid(tsid_json)
    df = pd.read_csv(
        input,
        dtype={col: str for col in tsid},
        usecols=[x, y, *tsid],
        parse_dates=[x],
        low_memory=False,
    )
    # Squash multiple txn/day into a single day.
    daily_agg_fun = "max" if kind.startswith("level") else "sum"
    df = df.groupby(by=[x, *tsid], as_index=False, group_keys=False).agg({y: daily_agg_fun})

    # Generate montages and individual images.
    mp = MontagePager(output_dir, page_size=100, savefig_kwargs={"transparent": transparent})
    title_01 = "#".join(tsid)

    # Decide whether to plot event curve (e.g., demand) vs level curve (e.g., price)
    fill_dt_kwargs: Dict[str, Any] = {}
    if kind.startswith("level"):
        fill_dt_kwargs = {
            "fillna_kwargs": dict(method="ffill"),
            "resample": "max",
        }

    # FIXME: to confirm removal
    # "Canvas" for individual plot
    # fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100, tight_layout=True)

    # Mapping file
    f = (output_dir / "mappings.csv").open("w")
    map_writer = csv.writer(f)
    map_writer.writerow(["individual", "title", "montage", "subplot", "row", "col"])

    for group_name, df_group in df.groupby(by=tsid):
        # Construct title
        title_02 = group_name if len(tsid) == 1 else "#".join(group_name)
        title = f"{title_01}: {title_02}"

        # Stretch each timeseries by filling-in the in-between gaps.
        ts = fill_dt(
            df_group.rename({x: "x"}, axis=1, copy=False),
            dates=pd.date_range(start, end, freq="D"),
            freq=freq,
            **fill_dt_kwargs,
        ).rename({"x": x}, axis=1, copy=False)
        # print(ts)
        # print(df_group)

        # Add plot to montage
        _plot(kind, df_group, ts, title=title, ax=mp.pop(title_02), x=x, y=y, grid=True)
    mp.savefig()  # Flush the last montage, if any.
    f.close()


def _plot(kind, ts_raw, ts_filled, title, **kwargs):
    ax = ts_filled.plot(title=title, **kwargs)
    ax.grid("on", which="minor", axis="x", linestyle=":")
    if kind == "level2":
        # Overlay the actual points on the plot as scatters
        ts_raw.plot(kind="scatter", **kwargs)
    return ax


def create_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # I/O matters
    parser.add_argument("input", nargs=1, type=Path2, help="Filename of input .csv.")
    parser.add_argument("--output-dir", default="output", type=Path2, help="Output directory")
    parser.add_argument(
        "--no-create-output-dir",
        dest="create_output_dir",
        action="store_false",
        default=True,
        help="Do not attempt to create OUTPUT_DIR",
    )

    # Data matters
    parser.add_argument("-x", "--x", type=str, default="x", help="Column name of x-axis aka timestamp (default: x)")
    parser.add_argument("-y", "--y", type=str, default="y", help="Column name of y-axis (default: y)")
    parser.add_argument(
        "-t",
        "--tsid",
        dest="tsid_json",
        type=str,
        help='Column name(s) of timeseries id, in JSON format (default: "tsid").',
    )
    parser.add_argument(
        "-k",
        "--kind",
        type=str,
        choices=["event", "level", "level2"],
        default="event",
        help=(
            'Whether this is "event" curve (e.g., demand curve) or "level" curve (e.g., price curve). '
            '"Level2" means level curve with actual point overlay.'
        ),
    )

    # Plot matters
    parser.add_argument(
        "-s",
        "--start",
        metavar="YYYY-MM-DD",
        type=str,
        default="2017-01-01",
        help="Start date used in the plot (default: 2017-01-01)",
    )
    parser.add_argument(
        "-e",
        "--end",
        metavar="YYYY-MM-DD",
        type=str,
        default=datetime.today().strftime("%Y-%m-%d"),
        help="End date used in the plot (default: today)",
    )
    parser.add_argument(
        "-f",
        "--freq",
        type=str,
        default="W",
        choices=["D", "W", "M"],
        help="Frequency used in the plot (default: W)",
    )
    parser.add_argument(
        "-r", "--transparent", action="store_true", default=False, help="Generate transparent backgrounds"
    )

    return parser


if __name__ == "__main__":
    args = create_argparse().parse_args()
    args.input = args.input[0]
    print(vars(args))
    main(**vars(args))
