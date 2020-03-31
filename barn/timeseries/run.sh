#!/bin/bash

set -e

# compute displacement
inputDirPath="/home/xing/hysds/stamps/data/apt-stamps-test/INSAR_20171004"
outputDirPath="."

python compute_displacement.py $inputDirPath $outputDirPath


# convert displacement from npy to pickle
inputFilePath="./displacement.npy"
outputFilePath="./displacement.pickle"

python convert_to_pickle.py $inputFilePath $outputFilePath

# plot to compare original (in npy) and interpolated (in pickle) displacement
npyFilePath=$inputFilePath
pickleFilePath=$outputFilePath
outputFilePath="./displacement.png"

python check_plot.py $npyFilePath $pickleFilePath $outputFilePath
