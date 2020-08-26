#!/bin/bash -xv

set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 workDirPath baseDirPath" >&2
    exit 1
fi

workDirPath=$1
baseDirPath=$2

timeStamp=$(date +%Y%m%dT%H%M%S.%NZ)
datasetName="landslide-prediction-"${timeStamp}

datasetDirPath=$workDirPath/${datasetName}

echo create dataset under $datasetDirPath

mkdir -p $datasetDirPath

echo "{}" > ${datasetDirPath}/${datasetName}.dataset.json
cp -pr ${baseDirPath}/ts_mintpy $datasetDirPath
cp -pr ${baseDirPath}/predictor $datasetDirPath
cp -pr ${baseDirPath}/plot $datasetDirPath
