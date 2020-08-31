import argparse
from typing import Any, Dict

import numpy as np
import pandas as pd

from ..pathlib import Path2, S3Path
from ._pdcheck import check_dtypes_distribution, check_possible_dtype, extract_str_values, is_fenced_str, read_csv

# region: pandas styling
padding = ("padding", "4px")
font = {
    "font-size": "10pt",
    "font-family": "sans-serif",
}
table_styles = [
    # Header
    dict(
        selector="thead tr th",
        props=[*font.items(), padding, ("border-bottom", "1px solid black"), ("background-color", "#99ccff")],
    ),
    # Row index
    dict(selector="tbody th", props=[*font.items(), padding]),
    # Whole table
    dict(selector="td", props=[padding]),
    dict(selector="tbody tr:nth-child(even)", props=[("background-color", "#e2e2e2")]),
    dict(selector="tbody tr:hover", props=[("background-color", "#ccffcc")]),
]
# endregion: pandas styling


def main(args: Dict[str, Any]):
    prefix = args["output_dir"]
    if args["create_output_dir"] and not isinstance(prefix, S3Path):
        prefix.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {}

    # This checks need dataframe loaded as strings (and without nan casting)
    results["guessed_dtypes"] = check_possible_dtype(read_csv(args["input"], as_str=True))

    # Rest checks fallback to pandas default behavior to autocast values to nearest dtype.
    df = read_csv(args["input"])
    results["dtypes_distribution"] = check_dtypes_distribution(df)
    results["fenced_str"] = extract_str_values(df, filter=lambda x: is_fenced_str(x))

    # Save all results as html and csv.
    for check_name, result_df in results.items():
        result_df.to_csv(prefix / f"{check_name}.csv", index=False)

        # Embed styles directly in the html -- https://stackoverflow.com/a/37760101
        #
        # NOTE: to separate styles into a .css, see
        # https://stackoverflow.com/questions/50807744/apply-css-class-to-pandas-dataframe-using-to-html
        # fmt: off
        html = (result_df
                    .style
                    .set_properties(**font)
                    .set_table_styles(table_styles)
                    .render()
        )
        # fmt: on
        with (prefix / f"{check_name}.html").open("w") as f:
            f.write(html)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs=1, type=Path2, help="Filename of input .csv.")
    parser.add_argument("--output-dir", default="output", type=Path2, help="Output directory")
    parser.add_argument(
        "--no-create-output-dir",
        dest="create_output_dir",
        action="store_false",
        default=True,
        help="Do not attempt to create OUTPUT_DIR",
    )
    args = parser.parse_args()
    args.input = args.input[0]
    main(vars(args))
