from typing import Dict

# Used for replacing invalid characters in tag values
_trs_tag = str.maketrans("$[]", "___")


def get_exec_dict(context, **kwargs) -> Dict[str, str]:
    """Convert Lambda execution context to a dictionary."""
    return {
        "lambda_req_id": context.aws_request_id,
        "lambda_log_group": context.log_group_name,
        "lambda_log_stream": context.log_stream_name.translate(_trs_tag),
        **kwargs,
    }
