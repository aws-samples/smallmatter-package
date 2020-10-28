from typing import Dict, List, Optional, cast

import boto3
from mypy_boto3_s3 import S3Client
from mypy_boto3_s3.type_defs import TagTypeDef

# Used for replacing invalid characters in tag values
trs_tag = str.maketrans("$[]", "___")


def lambda_exec_ctx_to_dict(context, **kwargs) -> Dict[str, str]:
    """Convert Lambda execution context to a dictionary."""
    return {
        "lambda_req_id": context.aws_request_id,
        "lambda_log_group": context.log_group_name,
        "lambda_log_stream": context.log_stream_name.translate(trs_tag),
        **kwargs,
    }


def add_tags(bucket: str, obj: str, tags: Dict[str, str], s3_client: Optional[S3Client] = None) -> List[TagTypeDef]:
    """Tag an S3 object with `tag_key = tag_value`."""
    # NOTE: `put_object_tagging()` overwrites existing tags set with a new set.
    # Hence, we do a write-after-read of tag sets to append new tags.

    if s3_client is None:
        s3_client = boto3.session.Session().client("s3")

    # Fetch existing tags. For the format of the response message, see
    # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#S3.Client.get_object_tagging
    existing_tags: List[TagTypeDef] = s3_client.get_object_tagging(Bucket=bucket, Key=obj)["TagSet"]

    # If there's existing tag, then we have to remove the old one, otherwise
    # put_object_tagging() complains about multiple tags with the same key.
    existing_tags2 = [d for d in existing_tags if d["Key"] not in tags]

    # Append new tag to existing tags
    new_tags = existing_tags2 + [{"Key": tag_key, "Value": cast(str, tag_value)} for tag_key, tag_value in tags.items()]

    # Put new tags to the S3 object.
    s3_client.put_object_tagging(Bucket=bucket, Key=obj, Tagging={"TagSet": new_tags})

    return new_tags
