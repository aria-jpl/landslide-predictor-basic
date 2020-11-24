#!/bin/bash

set -e

#if [ "$#" -ne 1 ]; then
#    echo "Usage: $0 dummy" >&2
#    exit 1
#fi

date

mintpyOutputDirPath=../ts_mintpy/input
#predictorInputDirPath=../predictor/input
predictorInputDirPath=../ts_mintpy/output
predictorOutputDirPath=../predictor/output
outputFilePath=./result.png

nice -20 \
    conda run -n python368 \
    python3.6 ./mintpy_ps.py \
    $mintpyOutputDirPath \
    $predictorInputDirPath \
    $predictorOutputDirPath \
    $outputFilePath

date
