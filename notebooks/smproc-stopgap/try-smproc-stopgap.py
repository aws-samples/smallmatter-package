from argparse import ArgumentParser
from pydoc import locate
from typing import Type, cast
from urllib.parse import urlparse

from loguru import logger
from sagemaker.estimator import Framework
from sagemaker.processing import ProcessingOutput

from smallmatter.pathlib import S3Path
from smallmatter.sm import FrameworkProcessor


def main(
    role: str,
    s3_prefix: S3Path,
    cls: Type[Framework],
    framework_version: str,
    py_version: str,
    separate_output: bool,
):
    """Submit a framework processing job."""
    processor = FrameworkProcessor(
        cls,
        framework_version=framework_version,
        py_version=py_version,
        role=role,
        instance_count=1,
        instance_type="ml.m5.large",
        s3_prefix=str(s3_prefix),
    )
    logger.info("Container uri: {}", processor.image_uri)

    # Whether processor output follows SageMaker training's style where output
    # goes under s3://..../jobname/output/.
    if separate_output:
        job_name = None
        output_dst = f"{s3_prefix}/output_always_overriden"
    else:
        job_name = processor._generate_current_job_name()
        output_dst = f"{s3_prefix}/{job_name}/output"

    processor.run(
        entry_point="processing.py",
        source_dir="./sourcedir",
        inputs=None,
        outputs=[
            ProcessingOutput(source="/opt/ml/processing/output", destination=output_dst),
        ],
        arguments=None,
        job_name=job_name,
    )


def s3ify(s) -> S3Path:
    """Convert a string (schema or schemaless) to an S3Path."""
    if urlparse(s).scheme == "s3":
        s = s[5:]
    return S3Path(s)


def classify(s) -> Type[Framework]:
    """Convert string to (hopefully) a class type."""
    cls = locate(s)
    if cls is None:
        raise ValueError(f"Cannot locate class {s}")
    return cast(type, cls)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--s3-prefix", type=s3ify, required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--framework_version", required=True)
    parser.add_argument("--py_version", default="py3")
    parser.add_argument("--separate-output", action="store_true")
    parser.add_argument("framework_cls", type=classify, metavar="SAGEMAKER_FRAMEWORK")

    args = parser.parse_args()
    logger.info("Script configuration: {}", vars(args))

    main(
        args.role,
        args.s3_prefix,
        args.framework_cls,
        args.framework_version,
        args.py_version,
        args.separate_output,
    )
