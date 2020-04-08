#!/bin/bash

set -e

# use this python env for h5py, etc.
python368="conda run -n python368 python"


# usage
programName=`basename $0`
if [ $# -ne 3 ]; then
    echo "Usage: $programName lonlatFilePath predictionFilePath outputFilePath" >&2
    exit 1
fi

eval "$(/opt/conda/bin/conda shell.bash hook)"

lonlatFilePath=$1
predictionFilePath=$2
outputFilePath=$3
$python368 plot_prediction.py $lonlatFilePath $predictionFilePath $outputFilePath
