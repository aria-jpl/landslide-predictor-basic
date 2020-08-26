#!/bin/bash

WORK_DIR=$(pwd)

BASE_PATH=$(dirname "${BASH_SOURCE}")
BASE_PATH=$(cd "${BASE_PATH}"; pwd)

#------
# do work at its own dir
# 20200823, xing
# this is a quick hack to go around file ownership issue in docker

mkdir -p $BASE_PATH/work

cp -avi $BASE_PATH/run.sh $BASE_PATH/work
cp -avi $BASE_PATH/get_param.py $BASE_PATH/work
cp -avi $BASE_PATH/create_dataset.sh $BASE_PATH/work

cp -avi $BASE_PATH/{ts_mintpy,predictor,plot} $BASE_PATH/work

(
cd $BASE_PATH/work

sh -xv ./run.sh
)

#------
# save result
(
echo "create dataset"

date

cd ${BASE_PATH}

sh -xv ./create_dataset.sh $WORK_DIR $BASE_PATH/work

date
)
