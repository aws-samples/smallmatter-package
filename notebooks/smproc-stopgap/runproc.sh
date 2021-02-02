#!/bin/bash

cd /opt/ml/processing/input/code/
tar -xzf payload/sourcedir.tar.gz

[[ -f 'requirements.txt' ]] && pip install -r requirements.txt

echo sagemaker_program=$sagemaker_program
python $sagemaker_program "$@"
