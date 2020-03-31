#!/bin/bash

whoami

echo $PATH

export LD_LIBRARY_PATH=/opt/conda/lib:/usr/lib:/usr/lib64:/usr/local/lib:$LD_LIBRARY_PATH
export PATH=/opt/conda/bin:$PATH

# https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-python.html
conda create --name python368 python=3.6.8 anaconda -y

echo aaaaaa

conda init bash

echo bbbbbb.0

pwd

echo bbbbbb

#export PS1="fakeAsLoginShell"
#source /home/ops/.bashrc
source ./conda-init.sh

echo cccccc

conda activate python368

echo dddddd

pip install -r ./requirements.txt
