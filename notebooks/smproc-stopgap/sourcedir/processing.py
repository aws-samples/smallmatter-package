import argparse
import os
from pathlib import Path

import dummy_util
import version_prober

if __name__ == "__main__":
    print("ENV_VARS:", os.environ)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-data-dir",
        type=Path,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/processing/output/"),
    )
    args = parser.parse_args()
    print("My arguments:", vars(args))

    # Some logic
    print("Helper function in file:", version_prober.__file__)
    print("Dummy function in file:", dummy_util.__file__)
    pkgs = version_prober.probe_ml_frameworks()

    # Write output
    with (args.output_data_dir / "processing.jsonl").open("w") as f:
        for mod in pkgs.values():
            f.write(f'{{"module": "{mod.__name__}", "version": "{mod.__version__}"}}\n')  # type: ignore

    print("Completed processing python entrypoint.")
