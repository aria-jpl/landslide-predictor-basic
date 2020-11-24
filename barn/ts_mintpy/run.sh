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
#sh ./fetch_from_s3.sh $s3Uri $inputDirPath

date

# ref: from giangi example
#python3 mintpy2pred.py -o ~/Volumes/giangidt/pred_mintpy/cali/pred_in/ -d ~/Volumes/giangidt/pred_mintpy/cali/inputs -c maskTempCoh_1.h5 -k -r .4 -s 0.01 -t timeseries_demErr_1.h5 -n pred_input_filt -f 10 -a 0.005

# convert to pickle
nice -20 \
    conda run -n python368 \
    python3.6 ./mintpy2pred.py \
    -d $inputDirPath \
    -o $outputDirPath \
    -c maskTempCoh.h5 \
    -k -r .4 -s 0.01 \
    -t timeseries_demErr.h5 \
    -n pred_input_filt \
    -f 10 -a 0.005
#    -e mask \
#    -k

date
