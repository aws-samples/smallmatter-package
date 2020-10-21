from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .pathlib import S3Path

# https://github.com/aws/sagemaker-python-sdk/blob/d8b3012c23fbccdcd1fda977ed9efa4507386a49/src/sagemaker/session.py#L45
NOTEBOOK_METADATA_FILE = "/opt/ml/metadata/resource-metadata.json"


def get_sm_execution_role() -> str:
    if Path(NOTEBOOK_METADATA_FILE).is_file():
        # Likely on SageMaker notebook instance.
        import sagemaker  # noqa

        return sagemaker.get_execution_role()
    else:
        # Unlikely on SageMaker notebook instance.
        # cf - https://github.com/aws/sagemaker-python-sdk/issues/300

        # Rely on botocore rather than boto3 for this function, to minimize
        # dependency on some environments where botocore exists, but not boto3.
        import botocore.session

        client = botocore.session.get_session().create_client("iam")
        response_roles = client.list_roles(
            PathPrefix="/",
            # Marker='string',
            MaxItems=999,
        )
        for role in response_roles["Roles"]:
            if role["RoleName"].startswith("AmazonSageMaker-ExecutionRole-"):
                # print('Resolved SageMaker IAM Role to: ' + str(role))
                return role["Arn"]
        raise Exception("Could not resolve what should be the SageMaker role to be used")


class PyTestHelpers:
    @staticmethod
    def import_from_file(name, fname):
        """Import a module given a specific file name.

        This is intended to run pytest tests on multiple SageMaker sourcedirs under a single git repo. For further
        example, see https://github.com/verdimrc/python-project-skeleton/blob/master/test/test_smep.py
        """
        import importlib.util

        spec = importlib.util.spec_from_file_location(name, fname)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


def get_model_and_output_tgz(train_job_name: str, sm: Optional[Any] = None) -> Tuple[S3Path, S3Path]:
    """[summary]

    Args:
        train_job_name (str): [description]
        sm (Optional[Any], optional): [description]. Defaults to None.

    Returns:
        Tuple[S3Path, S3Path]: (model.tar.gz, output.tar.gz).
    """
    model_tgz = get_model_tgz(train_job_name, sm)
    return (model_tgz, model_tgz.parent / "output.tar.gz")


def get_output_tgz(train_job_name: str, sm: Optional[Any] = None) -> S3Path:
    """Get the S3 path of the output.tar.gz produced by a completed train job.

    Args:
        train_job_name (str): Name of training job.
        sm (optional): boto3.client for sagemaker. Defaults to None.

    Returns:
        S3Path: S3 path to the output.tar.gz
    """
    model_tgz = get_model_tgz(train_job_name, sm)
    return model_tgz.parent / "output.tar.gz"


def get_model_tgz(train_job_name: str, sm: Optional[Any] = None) -> S3Path:
    """Get the S3 path of the model.tar.gz produced by a completed train job.

    Args:
        train_job_name (str): Name of training job.
        sm (optional): boto3.client for sagemaker. Defaults to None.

    Returns:
        S3Path: S3 path to the model.tar.gz
    """
    if sm is None:
        # NOTE: use boto3 instead of sagemaker sdk to minimize dependency.
        import boto3

        sm = boto3.client("sagemaker")

    resp: Dict[str, Any] = sm.describe_training_job(TrainingJobName=train_job_name)

    # Deal with not-completed job.
    job_status = resp["TrainingJobStatus"]
    if job_status != "Completed":
        raise ValueError(f"Training job {train_job_name} has status: {job_status}")

    # Given string s3://bucket/.../output/model.tar.gz, pick only "bucket/.../output/" (with trailing /)
    s3_prefix = resp["ModelArtifacts"]["S3ModelArtifacts"]
    return S3Path(s3_prefix[len("s3://") :])
