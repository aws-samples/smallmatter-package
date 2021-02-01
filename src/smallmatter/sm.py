from copy import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


try:
    import sagemaker
except ImportError:

    class FrameworkProcessor(object):
        def __init__(self, *args, **kwargs) -> None:
            raise NotImplementedError("Cannot find SageMaker Python SDK")


else:
    import logging

    from sagemaker.estimator import Framework
    from sagemaker.processing import ProcessingInput, ScriptProcessor
    from sagemaker.s3 import S3Uploader

    class FrameworkProcessor(ScriptProcessor):  # type: ignore
        logger = logging.getLogger("sagemaker")

        runproc_sh = """#!/bin/bash

cd /opt/ml/processing/input/code/
tar -xzf payload/sourcedir.tar.gz

[[ -f 'requirements.txt' ]] && pip install -r requirements.txt

#echo sagemaker_program=$sagemaker_program
#python $sagemaker_program "$@"

python {entry_point} "$@"
"""

        def __init__(
            self,
            estimator: Framework,
            role: str,
            instance_count: int,
            instance_type: str,
            s3_prefix: str,
            **kwargs,
        ):
            self.estimator = estimator
            if kwargs.get("base_job_name", None) is None:
                kwargs["base_job_name"] = self.estimator._framework_name
                if self.estimator._framework_name is None:
                    self.logger.warning("Framework name is None. Please check with the maintainer.")

            super().__init__(
                role=role,
                image_uri=estimator.training_image_uri(),
                command=["/bin/bash"],
                instance_count=instance_count,
                instance_type=instance_type,
                **kwargs,
            )

            self.s3_prefix = s3_prefix

        def run(
            self,
            entry_point: str,
            source_dir: Optional[str] = None,
            dependencies: Optional[Sequence[str]] = None,
            git_config: Optional[str] = None,
            inputs=None,
            outputs=None,
            arguments=None,
            wait=True,
            logs=True,
            job_name=None,
            experiment_config=None,
            kms_key=None,
        ):
            if job_name is None:
                job_name = self._generate_current_job_name()

            estimator = self._upload_payload(entry_point, source_dir, dependencies, git_config, job_name)
            inputs = self.patch_inputs_with_payload(estimator._hyperparameters["sagemaker_submit_directory"])

            # region: delete
            # FIXME: delete
            # Environment variables to tell the bootstrapper (i.e., runproc.sh) the filename of the
            # python entrypoint. Strictly speaking, runproc.sh needs only sagemaker_program, but
            # we'll inject all sagemaker_* just in case.
            # sm_env = {k: str(v) for k, v in estimator._hyperparameters.items() if k.startswith("sagemaker_")}
            # if self.env is None:  # type: ignore
            #    self.env = sm_env
            # else:
            #    self.env.update(sm_env)
            # endregion

            # Upload the bootstrapping code as s3://.../jobname/source/runproc.sh.
            s3_runproc_sh = S3Uploader.upload_string_as_file_body(
                self.runproc_sh.format(entry_point=entry_point),
                desired_s3_uri=f"{self.s3_prefix}/{job_name}/source/runproc.sh",
                sagemaker_session=self.sagemaker_session,
            )

            # Submit a processing job.
            super().run(
                code=s3_runproc_sh,
                inputs=inputs,
                outputs=outputs,
                arguments=arguments,
                wait=wait,
                logs=logs,
                job_name=job_name,
                experiment_config=experiment_config,
                kms_key=kms_key,
            )

        def _upload_payload(self, entry_point, source_dir, dependencies, git_config, job_name) -> Framework:
            # Be consistent with ScriptProcessor where self.run() can use different codes.
            estimator = copy(self.estimator)
            estimator.entry_point = entry_point
            estimator.source_dir = source_dir
            estimator.dependencies = dependencies
            estimator.git_config = git_config
            estimator._enable_network_isolation = False

            estimator._prepare_for_training(job_name=job_name)
            self.logger.info(
                "Uploaded %s to %s",
                estimator.source_dir,
                estimator._hyperparameters["sagemaker_submit_directory"],
            )

            return estimator

        def _patch_inputs_with_payload(self, inputs, s3_payload) -> List[ProcessingInput]:
            # ScriptProcessor job will download only s3://..../code/runproc.sh, hence we need to also
            # inject our s3://.../sourcedir.tar.gz.
            #
            # We'll follow the exact same mechanism that ScriptProcessor does, which is to inject the
            # S3 code artifact as a processing input with destination /opt/ml/processing/input/code/payload/.
            #
            # Unfortunately, as much as I'd like to put sourcedir.tar.gz to /opt/ml/processing/input/code/,
            # this cannot be done as this destination is already used by the ScriptProcessor for runproc.sh,
            # and the SDK vehemently refuses to have another input with the same destination.
            # - Note that the parameterized form of this path is available as ScriptProcessor._CODE_CONTAINER_BASE_PATH
            #   and ScriptProcessor._CODE_CONTAINER_INPUT_NAME.
            # - See: https://github.com/aws/sagemaker-python-sdk/blob/a7399455f5386d83ddc5cb15c0db00c04bd518ec/src/sagemaker/processing.py#L425-L426)
            if inputs is None:
                inputs = []
            inputs.append(ProcessingInput(source=s3_payload, destination="/opt/ml/processing/input/code/payload/"))
            return inputs
