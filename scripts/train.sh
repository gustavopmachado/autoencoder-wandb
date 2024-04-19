#!/bin/bash

# Parameters
ARCHITECTURE="AE"            # Architecture to be used in the model: AE, AAE, VAE
DATASET='MNIST'              # Name of the dataset in data/

# W&B
# export WANDB_SILENT=true
export WANDB_TAGS="train"

# Execute training
DIR=${PWD}
python ${DIR}/utils/run.py \
        --architecture $ARCHITECTURE \
        --dataset $DATASET \