import argparse
import os
from pathlib import Path

import mxnet

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

    with (args.output_data_dir / "processing.jsonl").open("w") as f:
        f.write(f'{{"module": "{mxnet.__name__}", "version": "{mxnet.__version__}"}}\n')

    print("Completed processing python entrypoint.")
