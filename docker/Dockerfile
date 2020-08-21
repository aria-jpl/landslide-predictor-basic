FROM hysds1/pge-base-conda-python368:20200818

LABEL maintainer="hysds"

USER ops

COPY ./barn/run.sh /home/ops

# timeseries

RUN mkdir -p ./timeseries

COPY ./barn/timeseries ./timeseries


# predictor

RUN mkdir -p ./predictor/{input,output}

COPY ./barn/predictor/pred_ps_package-2.tar.gz ./predictor
COPY ./barn/predictor/install.sh ./predictor
COPY ./barn/predictor/TFManager.py ./predictor
COPY ./barn/predictor/checkpoint ./predictor
COPY ./barn/predictor/run.sh ./predictor

COPY ./barn/predictor/ps.sample.pickle ./predictor

RUN cd ./predictor && \
    sh -xv ./install.sh && \
    rm ./pred_ps_package-2.tar.gz && \
    rm ./install.sh


# plot

RUN mkdir -p ./plot

COPY ./barn/plot ./plot


# Change to work directory
WORKDIR /home/ops

# Run shell
CMD ["/bin/bash", "--login"]
