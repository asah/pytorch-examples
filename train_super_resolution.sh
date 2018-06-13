#!/usr/bin/bash
# Entrypoint for Paperspace Job

# read/write Quilt packages from/to persistent storage
export QUILT_PRIMARY_PACKAGE_DIR='/storage/quilt_modules'

cd super_resolution

python main.py \
	--upscale_factor 3 \
	--batchSize 4 \
	--testBatchSize 100 \
	--nEpochs 10 \
	--lr 0.001 \
	--cuda
