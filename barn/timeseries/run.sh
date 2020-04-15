#!/bin/bash

set -e

# use this python env for h5py, etc.
python368="conda run -n python368 python"


# usage
programName=`basename $0`
if [ $# -ne 1 ]; then
    echo "Usage: $programName inputDirPath" >&2
    echo "       inputDirPath should be stamps run output dir such as ./INSAR_20171004" >&2
    exit 1
fi

eval "$(/opt/conda/bin/conda shell.bash hook)"

# compute displacement
inputDirPath=$1
outputDirPath="."
outputFileNamePrefix="ps"
#
$python368 compute_displacement.py $inputDirPath $outputDirPath $outputFileNamePrefix

# convert displacement from npy to pickle
prefix=$outputFileNamePrefix
for x in ./${prefix}.????.displacement.npy; do
    inputFilePath=${x}
    outputFilePath=${x%%.npy}.pickle
    $python368 convert_to_pickle.py $inputFilePath $outputFilePath
done

# plot to compare original (in npy) and interpolated (in pickle) displacement
for x in ./${prefix}.????.displacement.npy; do
    npyFilePath=${x}
    pickleFilePath=${x%%.npy}.pickle
    outputFilePath=${x%%.npy}.png
    $python368 check_plot.py $npyFilePath $pickleFilePath $outputFilePath
done
