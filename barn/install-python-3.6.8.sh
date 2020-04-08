#!/bin/bash

export LD_LIBRARY_PATH=/opt/conda/lib:/usr/lib:/usr/lib64:/usr/local/lib:$LD_LIBRARY_PATH
export PATH=/opt/conda/bin:$PATH

# https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-python.html
conda create --name python368 python=3.6.8 anaconda -y

conda init bash

source ./conda-init.sh

conda activate python368

pip install -r ./requirements.txt
