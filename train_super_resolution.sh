#!/usr/bin/bash
# USAGE: bash train_super_resolution.sh N_EPOCHS
# Entrypoint for Paperspace Job

# read/write Quilt packages from/to persistent storage
export QUILT_PRIMARY_PACKAGE_DIR='/storage/quilt_modules'

cd super_resolution

# Default to 10 epochs
$1 := 10

python main.py \
	--upscale_factor 3 \
	--batchSize 4 \
	--testBatchSize 100 \
	--nEpochs $1 \
	--lr 0.001 \
	--cuda
