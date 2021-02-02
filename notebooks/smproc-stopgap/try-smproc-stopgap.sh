#!/usr/bin/env bash

# Utility function to get script's directory (deal with Mac OSX quirkiness).
# This function is ambidextrous as it works on both Linux and OSX.
get_bin_dir() {
    local READLINK=readlink
    if [[ $(uname) == 'Darwin' ]]; then
        READLINK=greadlink
        if [ $(which greadlink) == '' ]; then
            echo '[ERROR] Mac OSX requires greadlink. Install with "brew install greadlink"' >&2
            exit 1
        fi
    fi

    local BIN_DIR=$(dirname "$($READLINK -f ${BASH_SOURCE[0]})")
    echo -n ${BIN_DIR}
}

declare -a ARGS=(
    --s3-prefix s3://bucket/prefix/sagemaker
    --role arn:aws:iam::111122223333:role/service-role/my-amazon-sagemaker-execution-role-1234
    --framework_version 1.6.0
    sagemaker.mxnet.estimator.MXNet
)

python $(get_bin_dir)/try-smproc-stopgap.py "${ARGS[@]}"
