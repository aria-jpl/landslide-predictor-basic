#!/bin/bash

set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 s3Uri" >&2
    exit 1
fi

s3Uri=$1

inputDirPath=./input
mkdir -p $inputDirPath

outputDirPath=./output
mkdir -p $outputDirPath

date

# fetch mintpy output from s3 to a local dir
sh ./fetch_from_s3.sh $s3Uri $inputDirPath

date

# convert to pickle
nice -20 \
    conda run -n python368 \
    python3.6 ./mintpy2pred.py \
    -d $inputDirPath \
    -c maskTempCoh.h5 \
    -t timeseries_demErr.h5 \
    -e mask \
    -o $outputDirPath \
    -k

date
