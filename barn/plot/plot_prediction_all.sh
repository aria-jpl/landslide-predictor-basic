#!/bin/bash

set -e

# use this python env for h5py, etc.
python368="conda run -n python368 python"


# usage
programName=`basename $0`
if [ $# -ne 4 ]; then
    echo "Usage: $programName fileNamePrefix lonlatDirPath predictionDirPath outputFilePath" >&2
    exit 1
fi

eval "$(/opt/conda/bin/conda shell.bash hook)"

fileNamePrefix=$1
lonlatDirPath=$2
predictionDirPath=$3
outputFilePath=$4
$python368 plot_prediction_all.py $fileNamePrefix $lonlatDirPath $predictionDirPath $outputFilePath
