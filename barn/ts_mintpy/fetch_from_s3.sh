#!/bin/bash

set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 remoteS3Uri localDirPath" >&2
    exit 1
fi

remoteS3Uri=$1
localDirPath=$2

date

echo fetch $remoteS3Uri to $localDirPath

aws s3 sync \
    $remoteS3Uri \
    $localDirPath

date
