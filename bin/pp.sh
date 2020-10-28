#!/usr/bin/env bash

# Make sure script is NOT sourced.
if [[ "${BASH_SOURCE[0]}" != $0 ]]; then
    echo "It appears that you sourced this script instead of executing it." >&2
    echo "Please set the executable bit 'chmod ugo+x <script>', then re-run." >&2
    return 1
fi

declare input=''         # filename.csv
declare output='output'  # output_dir
declare prefix='pp'      # prefix of .html reports

parse_args() {
    while [[ $# -gt 0 ]]; do
        key="$1"
        case $key in
        -h|--help)
            echo "Usage: $(basename ${BASH_SOURCE[0]}) <input_csv_filename> [--output output/prefix]"
            exit 0
            ;;
        --output)
            output=$2
            shift 2
            ;;
        --prefix)
            prefix=$2
            shift 2
            ;;
        *)
            input=$key
            shift
            ;;
        esac
    done

    if [[ $input == '' ]]; then
        echo 'Error: no input file' >&2
        echo "Usage: $(basename ${BASH_SOURCE[0]}) <input_csv_filename> [--output output/prefix]" >&2
        exit -1
    fi
}

parse_args "$@"

declare nproc=$(python -c 'import psutil; print(psutil.cpu_count(False), end="")')
declare -a config=(minimal default)

for i in "${config[@]}"; do
    declare -a CMD_LEFT=(
        /usr/bin/time pandas_profiling
        -s
        --pool_size $nproc
        --config_file config_${i}.yaml
        $input
        ${output}/${prefix}-${i}.html
    )

    declare -a CMD_RIGHT=(
        tee
        ${output}/${prefix}-${i}.txt
    )

    cmd="${CMD_LEFT[@]} 2>&1 | ${CMD_RIGHT[@]}"
    echo $cmd
    eval $cmd
done
