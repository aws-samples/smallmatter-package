#!/bin/bash

input=some/path/to/haha.csv
output=output/
nproc=$(python -c 'import psutil; print(psutil.cpu_count(False), end="")')


declare -a config=(minimal default)

for i in "${config[@]}"; do
    cmd="/usr/bin/time pandas_profiling -s --pool_size $nproc --config_file config_${i}.yaml $input ${output}-${i}.html 2>&1 | tee pp-${i}.txt"
    echo $cmd
    eval $cmd
done
