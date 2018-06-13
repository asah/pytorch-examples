#!/usr/bin/bash
# USAGE: bash train_super_resolution.sh N_EPOCHS
# Entrypoint for Paperspace Job

# read/write Quilt packages from/to persistent storage
export QUILT_PRIMARY_PACKAGE_DIR='/storage/quilt_modules'

cd super_resolution

N_EPOCHS=$1
# Default to 10
echo "Training for ${N_EPOCHS:=10} epochs\n"

python main.py \
	--upscale_factor 3 \
	--batchSize 4 \
	--testBatchSize 100 \
	--nEpochs $N_EPOCHS \
	--lr 0.001 \
	--cuda
