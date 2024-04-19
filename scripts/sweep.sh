#!/bin/bash

# Parameters
ARCHITECTURE="AE"             # Architecture to be used in the model: AE, AAE, VAE
BETAS=(0.9 0.999)             # Adam's Momentum
COUNT=50                      # Number of sweep config trials to try
DATASET='MNIST'               # Name of the dataset in data/
EPOCHS=50                     # Training epochs
LATENT=2                      # Number of features in the latent space
LOGFREQ=1                     # Wandb logging frequency

# W&B
export WANDB_SILENT=true
export WANDB_TAGS="sweep"

# Execute sweep
DIR=${PWD}
python ${DIR}/utils/sweep.py \
        --architecture $ARCHITECTURE \
        --batch $BATCH \
        --betas ${BETAS[@]} \
        --count $COUNT \
        --dataset $DATASET \
        --epochs $EPOCHS \
        --latent $LATENT \
        --logfreq $LOGFREQ