from pathlib import Path

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
