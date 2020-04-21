#!/bin/bash -xv

#set -e

# ------------------------------------------------------------------------------
# Automatically determines the full canonical path of where this script is
# located--regardless of what path this script is called from. (if available)
# ${BASH_SOURCE} works in both sourcing and execing the bash script.
# ${0} only works for when execing the bash script. ${0}==bash when sourcing.
BASE_PATH=$(dirname "${BASH_SOURCE}")
# convert potentially relative path to the full canonical path
BASE_PATH=$(cd "${BASE_PATH}"; pwd)
# get the name of the script
BASE_NAME=$(basename "${BASH_SOURCE}")
# ------------------------------------------------------------------------------

WORK_DIR=$(pwd)
echo "WORK_DIR:" $WORK_DIR

eval "$(/opt/conda/bin/conda shell.bash hook)"

echo "which conda"
which conda

#------
# prepare timeseries
(
echo "prepare timeseries"

date

cd $BASE_PATH/timeseries

inputDirPath=$(find -L "${WORK_DIR}" -type d| grep 's1-timeseries-ps-stamps-' | grep INSAR_)

echo "input dir path:", $inputDirPath

sh -xv ./run.sh $inputDirPath

# copy output to input dir of predictor
mv ./ps.????.displacement.pickle ${BASE_PATH}/predictor/input

date
)

#------
# run predictor
(
echo "run predictor"

date

cd $BASE_PATH/predictor

sh -xv ./run.sh

date
)

#------
# run plot
(
echo "run plot"

date

cd ${BASE_PATH}/plot

fileNamePrefix="ps"
lonlatDirPath=${BASE_PATH}/timeseries
predictionDirPath=${BASE_PATH}/predictor/output
outputFilePath=./${fileNamePrefix}.all.prediction.png
sh -xv ./plot_prediction_all.sh $fileNamePrefix $lonlatDirPath $predictionDirPath $outputFilePath

date
)

#------
# save result

TIMESERIES_DATASET=$(find -L . -type d | grep s1-timeseries|head -n 1)
TIMESERIES_DATASET=$(basename ${TIMESERIES_DATASET})
echo "TIMESERIES_DATASET:" ${TIMESERIES_DATASET}

LANDSLIDE_DATASET="landslides-${TIMESERIES_DATASET}"
echo "LANDSLIDE_DATASET:" ${LANDSLIDE_DATASET}
mkdir ${LANDSLIDE_DATASET}

cp ${TIMESERIES_DATASET}/${TIMESERIES_DATASET}.dataset.json  ${LANDSLIDE_DATASET}/${LANDSLIDE_DATASET}.dataset.json
cp -pr ${BASE_PATH}/timeseries ${LANDSLIDE_DATASET}
cp -pr ${BASE_PATH}/predictor ${LANDSLIDE_DATASET}
cp -pr ${BASE_PATH}/plot ${LANDSLIDE_DATASET}
