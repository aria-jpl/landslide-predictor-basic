#!/bin/bash

set -e

name=landslide-predictor-basic

docker run \
    --name ${name} \
    -v /home/xing/hysds/landslide/work/data:/var/data \
    -v /home/xing/hysds/stamps/data:/home/xing/hysds/stamps/data \
    -v /home/xing/hysds/landslide/timeseries:/home/xing/hysds/landslide/timeseries \
    --rm \
    -it \
    hysds/${name}:20200327 bash
