#!/bin/bash

set -e

# use this python env for h5py, etc.
python368="conda run -n python368 python"

# compute displacement
inputDirPath="/home/xing/hysds/stamps/data/apt-stamps-test/INSAR_20171004"
outputDirPath="."

$python368 compute_displacement.py $inputDirPath $outputDirPath

# convert displacement from npy to pickle
inputFilePath="./displacement.npy"
outputFilePath="./displacement.pickle"

$python368 convert_to_pickle.py $inputFilePath $outputFilePath

# plot to compare original (in npy) and interpolated (in pickle) displacement
npyFilePath=$inputFilePath
pickleFilePath=$outputFilePath
outputFilePath="./displacement.png"

$python368 check_plot.py $npyFilePath $pickleFilePath $outputFilePath
