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
    --cls_type subclass

    --s3-prefix s3://bucket/prefix/sagemaker
    --role arn:aws:iam::111122223333:role/service-role/my-amazon-sagemaker-execution-role-1234

    # Flags for MXNet processor
    --framework_version 1.6.0
    smallmatter.sm.MXNetProcessor

    # Flags to use MXNet processor in the forked-version of SageMaker SDK
    #--framework_version 1.6.0
    #sagemaker.mxnet.processing.MXNetProcessor


    # Flags for PyTorch processor
    #--framework_version 1.6.0
    #smallmatter.sm.PyTorchProcessor
)

cd $(get_bin_dir)
python ./try-smproc-stopgap.py "${ARGS[@]}" "$@"
