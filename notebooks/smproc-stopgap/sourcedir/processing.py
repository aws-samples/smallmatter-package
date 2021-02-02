import argparse
import importlib
import os
from pathlib import Path

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

    pkgs = dict()
    for module_name in ["mxnet", "sklearn", "torch", "tensorflow", "xgboost"]:
        try:
            pkgs[module_name] = importlib.import_module(module_name)
        except ModuleNotFoundError:
            pass

    with (args.output_data_dir / "processing.jsonl").open("w") as f:
        for mod in pkgs.values():
            f.write(f'{{"module": "{mod.__name__}", "version": "{mod.__version__}"}}\n')  # type: ignore

    print("Completed processing python entrypoint.")
