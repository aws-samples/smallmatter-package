from pathlib import Path
from time import sleep
from typing import Any, Dict, List, Optional, Tuple, Type

from botocore.credentials import AssumeRoleCredentialFetcher

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


def search_completed_processing_jobs_with_substring(
    substring: str,
    prefix: bool = True,
    sm: Optional[Any] = None,
    delay: float = 0.2,
) -> List[Dict[str, Any]]:
    """Get the processing jobs whose name contains `substring`.

    Profiler reports are not be included in the search results.

    Args:
        substring (str): Substring of processing job name.
        prefix (bool, optional): Whether substring must be the prefix. Defaults to True.
        sm (optional): boto3.client for sagemaker. Defaults to None.
        delay (float, optional): the delay (in seconds) between successive requests when SageMaker
            truncate search results. Defaults to 0.2 second.
    Returns:
        List: List of processing jobs. This corresponds to the concatenation of
            ``response["ProcessingJobSummaries"][0]`` returned by `boto3.sagemaker.list_processing_jobs()`, with
            ``response`` returned by `boto3.sagemaker.describe_processing_job()`. A new field is
            also created: "ProcessingTimeInSeconds".
    """
    if sm is None:
        # NOTE: use boto3 instead of sagemaker sdk to minimize dependency.
        import boto3
        from botocore.config import Config

        sm = boto3.client(
            service_name="sagemaker",
            config=Config(connect_timeout=5, read_timeout=60, retries={"max_attempts": 20, "mode": "adaptive"}),
        )

    # Pass-01: retrieve job names
    job_summaries = []

    list_params = dict(NameContains=substring, StatusEquals="Completed", MaxResults=100)

    while True:
        resp = sm.list_processing_jobs(**list_params)
        job_summaries.extend(resp["ProcessingJobSummaries"])
        next_token = resp.get("NextToken", None)
        if next_token is None:
            break
        list_params["NextToken"] = next_token
        sleep(delay)

    # Drop profiler reports
    job_summaries = [
        job_summary for job_summary in job_summaries if "-ProfilerReport-" not in job_summary["ProcessingJobName"]
    ]

    if prefix:
        job_summaries = [
            job_summary for job_summary in job_summaries if job_summary["ProcessingJobName"].startswith(substring)
        ]

    # Pass-02: for each job name, get the detail description.
    results = []
    for job_summary in job_summaries:
        resp = sm.describe_processing_job(ProcessingJobName=job_summary["ProcessingJobName"])
        resp["ProcessingTimeInSeconds"] = (resp["ProcessingEndTime"] - resp["ProcessingStartTime"]).total_seconds()
        results.append(resp)
        sleep(delay)

    # Pass-03: add job duration

    return results


def search_completed_training_jobs_with_substring(
    substring: str,
    prefix: bool = True,
    sm: Optional[Any] = None,
    delay: float = 0.2,
) -> List[Dict[str, Any]]:
    """Get the training jobs whose name contains `substring` either as prefix or anywhere.

    Args:
        prefix (str): Substring of training job name.
        sm (optional): boto3.client for sagemaker. Defaults to None.
        delay (float, optional): the delay (in seconds) between successive requests when search results exceed 100.
            Defaults to 0.2 second.
    Returns:
        List: List of training jobs. This corresponds to the ``response["Results"][0]["TrainingJob"]`` returned by
            `boto3.sagemaker.search()`.
    """
    if sm is None:
        # NOTE: use boto3 instead of sagemaker sdk to minimize dependency.
        import boto3

        sm = boto3.client(service_name="sagemaker")

    search_params = {
        "Resource": "TrainingJob",
        "SearchExpression": {
            "Filters": [
                {
                    "Name": "TrainingJobName",
                    "Operator": "Contains",
                    "Value": substring,
                },
                {
                    "Name": "TrainingJobStatus",
                    "Operator": "Equals",
                    "Value": "Completed",
                },
            ]
        },
    }

    results = []
    while True:
        resp = sm.search(**search_params)
        results.extend(resp["Results"])
        next_token = resp.get("NextToken", None)
        if next_token is None:
            break
        search_params["NextToken"] = next_token
        sleep(delay)

    results = [result["TrainingJob"] for result in results]
    if prefix:
        results = [result for result in results if result["TrainingJobName"].startswith(substring)]
    return results


try:
    import sagemaker
except ImportError:

    class FrameworkProcessor(object):
        def __init__(self, *args, **kwargs) -> None:
            raise NotImplementedError("Cannot find SageMaker Python SDK")


else:
    import logging

    from sagemaker.estimator import Framework
    from sagemaker.mxnet.estimator import MXNet
    from sagemaker.network import NetworkConfig
    from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
    from sagemaker.pytorch.estimator import PyTorch
    from sagemaker.s3 import S3Uploader
    from sagemaker.session import Session
    from sagemaker.sklearn.estimator import SKLearn
    from sagemaker.tensorflow.estimator import TensorFlow
    from sagemaker.xgboost.estimator import XGBoost

    class FrameworkProcessor(ScriptProcessor):  # type: ignore
        """Handles Amazon SageMaker processing tasks for jobs using a machine learning framework."""

        logger = logging.getLogger("sagemaker")

        runproc_sh = """#!/bin/bash

cd /opt/ml/processing/input/code/
tar -xzf payload/sourcedir.tar.gz

[[ -f 'requirements.txt' ]] && pip install -r requirements.txt

python {entry_point} "$@"
"""
        # Added new (kw)args for estimator. The rest are from ScriptProcessor with same defaults.
        def __init__(
            self,
            estimator_cls: Type[Framework],  # New arg
            framework_version: str,  # New arg
            s3_prefix: str,  # New arg
            role: str,
            instance_count: int,
            instance_type: str,
            py_version: str = "py3",  # New kwarg
            image_uri: Optional[str] = None,
            volume_size_in_gb: int = 30,
            volume_kms_key: Optional[str] = None,
            output_kms_key: Optional[str] = None,
            max_runtime_in_seconds: Optional[int] = None,
            base_job_name: Optional[str] = None,
            sagemaker_session: Optional[Session] = None,
            env: Optional[Dict[str, str]] = None,
            tags: Optional[List[Dict[str, Any]]] = None,
            network_config: Optional[NetworkConfig] = None,
        ):
            """Initializes a ``FrameworkProcessor`` instance.

            The ``FrameworkProcessor`` handles Amazon SageMaker Processing tasks for jobs
            using a machine learning framework, which allows for a set of Python scripts
            to be run as part of the Processing Job.

            Args:
                estimator_cls (type): A subclass of ``Framework`` estimator
                framework_version (str): The version of the framework
                s3_prefix (str): The S3 prefix URI where custom code will be
                    uploaded - don't include a trailing slash since a string prepended
                    with a "/" is appended to ``s3_prefix``. The code file uploaded to S3
                    is 's3_prefix/job-name/source/sourcedir.tar.gz'.
                role (str): An AWS IAM role name or ARN. Amazon SageMaker Processing uses
                    this role to access AWS resources, such as data stored in Amazon S3.
                instance_count (int): The number of instances to run a processing job with.
                instance_type (str): The type of EC2 instance to use for processing, for
                    example, 'ml.c4.xlarge'.
                py_version (str): Python version you want to use for executing your
                    model training code. One of 'py2' or 'py3'. Defaults to 'py3'. Value
                    is ignored when ``image_uri`` is provided.
                image_uri (str): The URI of the Docker image to use for the
                    processing jobs.
                volume_size_in_gb (int): Size in GB of the EBS volume
                    to use for storing data during processing (default: 30).
                volume_kms_key (str): A KMS key for the processing volume (default: None).
                output_kms_key (str): The KMS key ID for processing job outputs (default: None).
                max_runtime_in_seconds (int): Timeout in seconds (default: None).
                    After this amount of time, Amazon SageMaker terminates the job,
                    regardless of its current status. If `max_runtime_in_seconds` is not
                    specified, the default value is 24 hours.
                base_job_name (str): Prefix for processing name. If not specified,
                    the processor generates a default job name, based on the
                    processing image name and current timestamp.
                sagemaker_session (:class:`~sagemaker.session.Session`):
                    Session object which manages interactions with Amazon SageMaker and
                    any other AWS services needed. If not specified, the processor creates
                    one using the default AWS configuration chain.
                env (dict[str, str]): Environment variables to be passed to
                    the processing jobs (default: None).
                tags (list[dict]): List of tags to be passed to the processing job
                    (default: None). For more, see
                    https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
                network_config (:class:`~sagemaker.network.NetworkConfig`):
                    A :class:`~sagemaker.network.NetworkConfig`
                    object that configures network isolation, encryption of
                    inter-container traffic, security group IDs, and subnets.
            """
            self.estimator_cls = estimator_cls
            self.framework_version = framework_version
            self.py_version = py_version

            image_uri, base_job_name = self._pre_init_normalization(
                instance_count, instance_type, image_uri, base_job_name
            )

            super().__init__(
                role=role,
                image_uri=image_uri,
                command=["/bin/bash"],
                instance_count=instance_count,
                instance_type=instance_type,
                volume_size_in_gb=volume_size_in_gb,
                volume_kms_key=volume_kms_key,
                output_kms_key=output_kms_key,
                max_runtime_in_seconds=max_runtime_in_seconds,
                base_job_name=base_job_name,
                sagemaker_session=sagemaker_session,
                env=env,
                tags=tags,
                network_config=network_config,
            )

            self.s3_prefix = s3_prefix

        def _pre_init_normalization(
            self,
            instance_count: int,
            instance_type: str,
            image_uri: Optional[str] = None,
            base_job_name: Optional[str] = None,
        ) -> Tuple[str, str]:
            # Normalize base_job_name
            if base_job_name is None:
                base_job_name = self.estimator_cls._framework_name
                if base_job_name is None:
                    self.logger.warning("Framework name is None. Please check with the maintainer.")
                    base_job_name = str(base_job_name)  # Keep mypy happy.

            # Normalize image uri.
            if image_uri is None:
                # Estimator used only to probe image uri, so can get away with some dummy values.
                est = self.estimator_cls(
                    framework_version=self.framework_version,
                    instance_type=instance_type,
                    py_version=self.py_version,
                    image_uri=image_uri,
                    entry_point="",
                    role="",
                    enable_network_isolation=False,
                    instance_count=instance_count,
                )
                image_uri = est.training_image_uri()

            return image_uri, base_job_name

        def run(
            self,
            entry_point: str,
            source_dir: Optional[str],
            dependencies: Optional[List[str]] = None,
            git_config: Optional[Dict[str, str]] = None,
            inputs: Optional[List[ProcessingInput]] = None,
            outputs: Optional[List[ProcessingOutput]] = None,
            arguments: Optional[List[str]] = None,
            wait: bool = True,
            logs: bool = True,
            job_name: Optional[str] = None,
            experiment_config: Optional[Dict[str, str]] = None,
            kms_key: Optional[str] = None,
        ):
            """Runs a processing job.

            Args:
                entrypoint (str): Path (absolute or relative) to the local Python source
                    file which should be executed as the entry point to training. If
                    ``source_dir`` is specified, then ``entry_point`` must point to a file
                    located at the root of ``source_dir``.
                source_dir (str): Path (absolute, relative or an S3 URI) to a directory
                    with any other training source code dependencies aside from the entry
                    point file (default: None). If ``source_dir`` is an S3 URI, it must
                    point to a tar.gz file. Structure within this directory are preserved
                    when training on Amazon SageMaker.
                dependencies (list[str]): A list of paths to directories (absolute
                    or relative) with any additional libraries that will be exported
                    to the container (default: []). The library folders will be
                    copied to SageMaker in the same folder where the entrypoint is
                    copied. If 'git_config' is provided, 'dependencies' should be a
                    list of relative locations to directories with any additional
                    libraries needed in the Git repo.
                git_config (dict[str, str]): Git configurations used for cloning
                    files, including ``repo``, ``branch``, ``commit``,
                    ``2FA_enabled``, ``username``, ``password`` and ``token``. The
                    ``repo`` field is required. All other fields are optional.
                    ``repo`` specifies the Git repository where your training script
                    is stored. If you don't provide ``branch``, the default value
                    'master' is used. If you don't provide ``commit``, the latest
                    commit in the specified branch is used. .. admonition:: Example

                        The following config:

                        >>> git_config = {'repo': 'https://github.com/aws/sagemaker-python-sdk.git',
                        >>>               'branch': 'test-branch-git-config',
                        >>>               'commit': '329bfcf884482002c05ff7f44f62599ebc9f445a'}

                        results in cloning the repo specified in 'repo', then
                        checkout the 'master' branch, and checkout the specified
                        commit.

                    ``2FA_enabled``, ``username``, ``password`` and ``token`` are
                    used for authentication. For GitHub (or other Git) accounts, set
                    ``2FA_enabled`` to 'True' if two-factor authentication is
                    enabled for the account, otherwise set it to 'False'. If you do
                    not provide a value for ``2FA_enabled``, a default value of
                    'False' is used. CodeCommit does not support two-factor
                    authentication, so do not provide "2FA_enabled" with CodeCommit
                    repositories.

                    For GitHub and other Git repos, when SSH URLs are provided, it
                    doesn't matter whether 2FA is enabled or disabled; you should
                    either have no passphrase for the SSH key pairs, or have the
                    ssh-agent configured so that you will not be prompted for SSH
                    passphrase when you do 'git clone' command with SSH URLs. When
                    HTTPS URLs are provided: if 2FA is disabled, then either token
                    or username+password will be used for authentication if provided
                    (token prioritized); if 2FA is enabled, only token will be used
                    for authentication if provided. If required authentication info
                    is not provided, python SDK will try to use local credentials
                    storage to authenticate. If that fails either, an error message
                    will be thrown.

                    For CodeCommit repos, 2FA is not supported, so '2FA_enabled'
                    should not be provided. There is no token in CodeCommit, so
                    'token' should not be provided too. When 'repo' is an SSH URL,
                    the requirements are the same as GitHub-like repos. When 'repo'
                    is an HTTPS URL, username+password will be used for
                    authentication if they are provided; otherwise, python SDK will
                    try to use either CodeCommit credential helper or local
                    credential storage for authentication.
                inputs (list[:class:`~sagemaker.processing.ProcessingInput`]): Input files for
                    the processing job. These must be provided as
                    :class:`~sagemaker.processing.ProcessingInput` objects (default: None).
                outputs (list[:class:`~sagemaker.processing.ProcessingOutput`]): Outputs for
                    the processing job. These can be specified as either path strings or
                    :class:`~sagemaker.processing.ProcessingOutput` objects (default: None).
                arguments (list[str]): A list of string arguments to be passed to a
                    processing job (default: None).
                wait (bool): Whether the call should wait until the job completes (default: True).
                logs (bool): Whether to show the logs produced by the job.
                    Only meaningful when wait is True (default: True).
                job_name (str): Processing job name. If not specified, the processor generates
                    a default job name, based on the base job name and current timestamp.
                experiment_config (dict[str, str]): Experiment management configuration.
                    Dictionary contains three optional keys:
                    'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.
                kms_key (str): The ARN of the KMS key that is used to encrypt the
                    user code file (default: None).
            """
            if job_name is None:
                job_name = self._generate_current_job_name()

            estimator = self._upload_payload(entry_point, source_dir, dependencies, git_config, job_name)
            inputs = self._patch_inputs_with_payload(inputs, estimator._hyperparameters["sagemaker_submit_directory"])

            # Upload the bootstrapping code as s3://.../jobname/source/runproc.sh.
            s3_runproc_sh = S3Uploader.upload_string_as_file_body(
                self.runproc_sh.format(entry_point=entry_point),
                desired_s3_uri=f"{self.s3_prefix}/{job_name}/source/runproc.sh",
                sagemaker_session=self.sagemaker_session,
            )
            self.logger.info("runproc.sh uploaded to", s3_runproc_sh)

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

        def _upload_payload(
            self,
            entry_point: str,
            source_dir: Optional[str],
            dependencies: Optional[List[str]],
            git_config: Optional[Dict[str, str]],
            job_name: str,
        ) -> Framework:
            # A new estimator instance is required, because each call to ScriptProcessor.run() can
            # use different codes.
            estimator = self.estimator_cls(
                entry_point=entry_point,
                source_dir=source_dir,
                dependencies=dependencies,
                git_config=git_config,
                framework_version=self.framework_version,
                py_version=self.py_version,
                code_location=self.s3_prefix,  # Estimator will use <code_location>/jobname/output/source.tar.gz
                enable_network_isolation=False,  # If true, estimator uploads to input channel. Not what we want!
                image_uri=self.image_uri,  # The image uri is already normalized by this point.
                role=self.role,
                instance_type=self.instance_type,
                instance_count=self.instance_count,
                sagemaker_session=self.sagemaker_session,
                debugger_hook_config=False,
                disable_profiler=True,
            )

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
            # and the SDK does not allow another input with the same destination.
            # - Note that the parameterized form of this path is available as ScriptProcessor._CODE_CONTAINER_BASE_PATH
            #   and ScriptProcessor._CODE_CONTAINER_INPUT_NAME.
            # - See: https://github.com/aws/sagemaker-python-sdk/blob/a7399455f5386d83ddc5cb15c0db00c04bd518ec/src/sagemaker/processing.py#L425-L426)
            if inputs is None:
                inputs = []
            inputs.append(ProcessingInput(source=s3_payload, destination="/opt/ml/processing/input/code/payload/"))
            return inputs

    class MXNetProcessor(FrameworkProcessor):
        """Handles Amazon SageMaker processing tasks for jobs using MXNet containers."""

        estimator_cls = MXNet

        def __init__(
            self,
            framework_version: str,  # New arg
            s3_prefix: str,  # New arg
            role: str,
            instance_count: int,
            instance_type: str,
            py_version: str = "py3",  # New kwarg
            image_uri: Optional[str] = None,
            volume_size_in_gb: int = 30,
            volume_kms_key: Optional[str] = None,
            output_kms_key: Optional[str] = None,
            max_runtime_in_seconds: Optional[int] = None,
            base_job_name: Optional[str] = None,
            sagemaker_session: Optional[Session] = None,
            env: Optional[Dict[str, str]] = None,
            tags: Optional[List[Dict[str, Any]]] = None,
            network_config: Optional[NetworkConfig] = None,
        ):
            """This processor executes a Python script in a managed MXNet execution environment.

            Unless ``image_uri`` is specified, the MXNet environment is an
            Amazon-built Docker container that executes functions defined in the supplied
            ``entry_point`` Python script.

            The arguments have the exact same meaning as in ``FrameworkProcessor``.

            .. tip::

                You can find additional parameters for initializing this class at
                :class:`~smallmatter.ds.FrameworkProcessor`.
            """
            super().__init__(
                self.estimator_cls,
                framework_version,
                s3_prefix,
                role,
                instance_count,
                instance_type,
                py_version,
                image_uri,
                volume_size_in_gb,
                volume_kms_key,
                output_kms_key,
                max_runtime_in_seconds,
                base_job_name,
                sagemaker_session,
                env,
                tags,
                network_config,
            )

    class PyTorchProcessor(FrameworkProcessor):
        """Handles Amazon SageMaker processing tasks for jobs using PyTorch containers."""

        estimator_cls = PyTorch

        def __init__(
            self,
            framework_version: str,  # New arg
            s3_prefix: str,  # New arg
            role: str,
            instance_count: int,
            instance_type: str,
            py_version: str = "py3",  # New kwarg
            image_uri: Optional[str] = None,
            volume_size_in_gb: int = 30,
            volume_kms_key: Optional[str] = None,
            output_kms_key: Optional[str] = None,
            max_runtime_in_seconds: Optional[int] = None,
            base_job_name: Optional[str] = None,
            sagemaker_session: Optional[Session] = None,
            env: Optional[Dict[str, str]] = None,
            tags: Optional[List[Dict[str, Any]]] = None,
            network_config: Optional[NetworkConfig] = None,
        ):
            """This processor executes a Python script in a PyTorch execution environment.

            Unless ``image_uri`` is specified, the PyTorch environment is an
            Amazon-built Docker container that executes functions defined in the supplied
            ``entry_point`` Python script.

            The arguments have the exact same meaning as in ``FrameworkProcessor``.

            .. tip::

                You can find additional parameters for initializing this class at
                :class:`~smallmatter.ds.FrameworkProcessor`.
            """
            super().__init__(
                self.estimator_cls,
                framework_version,
                s3_prefix,
                role,
                instance_count,
                instance_type,
                py_version,
                image_uri,
                volume_size_in_gb,
                volume_kms_key,
                output_kms_key,
                max_runtime_in_seconds,
                base_job_name,
                sagemaker_session,
                env,
                tags,
                network_config,
            )

    class SKLearnProcessorAlt(FrameworkProcessor):
        """Handles Amazon SageMaker processing tasks for jobs using scikit-learn containers."""

        estimator_cls = SKLearn

        def __init__(
            self,
            framework_version: str,  # New arg
            s3_prefix: str,  # New arg
            role: str,
            instance_count: int,
            instance_type: str,
            py_version: str = "py3",  # New kwarg
            image_uri: Optional[str] = None,
            volume_size_in_gb: int = 30,
            volume_kms_key: Optional[str] = None,
            output_kms_key: Optional[str] = None,
            max_runtime_in_seconds: Optional[int] = None,
            base_job_name: Optional[str] = None,
            sagemaker_session: Optional[Session] = None,
            env: Optional[Dict[str, str]] = None,
            tags: Optional[List[Dict[str, Any]]] = None,
            network_config: Optional[NetworkConfig] = None,
        ):
            """This processor executes a Python script in a scikit-learn execution environment.

            This class has an 'Alt' suffix to denote it as an alternative to built-in
            ``sagemaker.sklearn.processing.SKLearnProcessor``.

            Unless ``image_uri`` is specified, the scikit-learn environment is an
            Amazon-built Docker container that executes functions defined in the supplied
            ``entry_point`` Python script.

            The arguments have the exact same meaning as in ``FrameworkProcessor``.

            .. tip::

                You can find additional parameters for initializing this class at
                :class:`~smallmatter.ds.FrameworkProcessor`.
            """
            super().__init__(
                self.estimator_cls,
                framework_version,
                s3_prefix,
                role,
                instance_count,
                instance_type,
                py_version,
                image_uri,
                volume_size_in_gb,
                volume_kms_key,
                output_kms_key,
                max_runtime_in_seconds,
                base_job_name,
                sagemaker_session,
                env,
                tags,
                network_config,
            )

    class TensorFlowProcessor(FrameworkProcessor):
        """Handles Amazon SageMaker processing tasks for jobs using TensorFlow containers."""

        estimator_cls = TensorFlow

        def __init__(
            self,
            framework_version: str,  # New arg
            s3_prefix: str,  # New arg
            role: str,
            instance_count: int,
            instance_type: str,
            py_version: str = "py3",  # New kwarg
            image_uri: Optional[str] = None,
            volume_size_in_gb: int = 30,
            volume_kms_key: Optional[str] = None,
            output_kms_key: Optional[str] = None,
            max_runtime_in_seconds: Optional[int] = None,
            base_job_name: Optional[str] = None,
            sagemaker_session: Optional[Session] = None,
            env: Optional[Dict[str, str]] = None,
            tags: Optional[List[Dict[str, Any]]] = None,
            network_config: Optional[NetworkConfig] = None,
        ):
            """This processor executes a Python script in a TensorFlow execution environment.

            Unless ``image_uri`` is specified, the TensorFlow environment is an
            Amazon-built Docker container that executes functions defined in the supplied
            ``entry_point`` Python script.

            The arguments have the exact same meaning as in ``FrameworkProcessor``.

            .. tip::

                You can find additional parameters for initializing this class at
                :class:`~smallmatter.ds.FrameworkProcessor`.
            """
            super().__init__(
                self.estimator_cls,
                framework_version,
                s3_prefix,
                role,
                instance_count,
                instance_type,
                py_version,
                image_uri,
                volume_size_in_gb,
                volume_kms_key,
                output_kms_key,
                max_runtime_in_seconds,
                base_job_name,
                sagemaker_session,
                env,
                tags,
                network_config,
            )

    class XGBoostEstimator(FrameworkProcessor):
        """Handles Amazon SageMaker processing tasks for jobs using XGBoost containers."""

        estimator_cls = XGBoost

        def __init__(
            self,
            framework_version: str,  # New arg
            s3_prefix: str,  # New arg
            role: str,
            instance_count: int,
            instance_type: str,
            py_version: str = "py3",  # New kwarg
            image_uri: Optional[str] = None,
            volume_size_in_gb: int = 30,
            volume_kms_key: Optional[str] = None,
            output_kms_key: Optional[str] = None,
            max_runtime_in_seconds: Optional[int] = None,
            base_job_name: Optional[str] = None,
            sagemaker_session: Optional[Session] = None,
            env: Optional[Dict[str, str]] = None,
            tags: Optional[List[Dict[str, Any]]] = None,
            network_config: Optional[NetworkConfig] = None,
        ):
            """This processor executes a Python script in an XGBoost execution environment.

            Unless ``image_uri`` is specified, the XGBoost environment is an Amazon-built
            Docker container that executes functions defined in the supplied ``entry_point``
            Python script.

            The arguments have the exact same meaning as in ``FrameworkProcessor``.

            .. tip::

                You can find additional parameters for initializing this class at
                :class:`~smallmatter.ds.FrameworkProcessor`.
            """
            super().__init__(
                self.estimator_cls,
                framework_version,
                s3_prefix,
                role,
                instance_count,
                instance_type,
                py_version,
                image_uri,
                volume_size_in_gb,
                volume_kms_key,
                output_kms_key,
                max_runtime_in_seconds,
                base_job_name,
                sagemaker_session,
                env,
                tags,
                network_config,
            )
