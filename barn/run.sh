#!/bin/bash

set -e

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

# prepare timeseries
(
date

cd $BASE_PATH/timeseries

#inputDirPath=/home/xing/hysds/stamps/data/apt-stamps-test/INSAR_20171004
#inputDirPath=/home/xing/hysds/stamps/data/apt-stamps-test/INSAR_20171004
#inputDirPath=$( $BASE_PATH/s1-timeseries-ps-stamps-*/INSAR_*)
####find "${BASE_PATH}/s1-timeseries-ps-stamps-*" -type d -name 's1-timeseries-ps-stamps-*' | grep INSAR_ | while read DIR; do
#find "${BASE_PATH}" -type d -name 's1-timeseries-ps-stamps-*' | grep INSAR_ | while read DIR; do
#    INSAR_DATE_DIR=${DIR}
#    break
#done
inputDirPath=$(find -L "${BASE_PATH}" -type d| grep 's1-timeseries-ps-stamps-' | grep INSAR_)

echo "++++++++++=", $inputDirPath

sh ./run.sh $inputDirPath

# copy output to input dir of predictor
pickleFileName=displacement.pickle
mv ./$pickleFileName ${BASE_PATH}/predictor/input

date
)

# run predictor
(
date

cd $BASE_PATH/predictor

sh ./run.sh

date
)

#TIMESTAMP=$(date +%Y%m%dT%H%M%S%Z)
#DATASET="landslide-predictor-results-${TIMESTAMP}"
#mkdir ${DATASET}
#mv /home/ops/{timeseries,predictor} ${DATASET}
#cd ${BASE_PATH}

#TIMESERIES_DATASET="$(ls s1-timeseries-ps-stamps*) | head -1"
TIMESERIES_DATASET=$(find -L . -type d | grep s1-timeseries|head -n 1)
TIMESERIES_DATASET=$(basename ${TIMESERIES_DATASET})
echo "+++++++++++++=" ${TIMESERIES_DATASET}
LANDSLIDE_DATASET="landslides-${TIMESERIES_DATASET}"
echo ${LANDSLIDE_DATASET}
mkdir ${LANDSLIDE_DATASET}
cp ${TIMESERIES_DATASET}/${TIMESERIES_DATASET}.dataset.json  ${LANDSLIDE_DATASET}/${LANDSLIDE_DATASET}.dataset.json
#mv ${BASE_PATH}/{timeseries,predictor} ${LANDSLIDE_DATASET}
mv ${BASE_PATH}/timeseries ${LANDSLIDE_DATASET}
mv ${BASE_PATH}/predictor ${LANDSLIDE_DATASET}
