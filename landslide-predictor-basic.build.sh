#!/bin/bash

set -e

name=landslide-predictor-basic
context=.

#    --no-cache \
docker build ${context} \
    --file ${context}/${name}/Dockerfile \
    --tag hysds/${name}:20200327
