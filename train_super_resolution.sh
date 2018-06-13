#!/usr/bin/bash
# USAGE: bash train_super_resolution.sh N_EPOCHS
# Entrypoint for Paperspace Job

# requirements
pip install -r super_resolution/requirements.txt
# read/write Quilt packages from/to persistent storage
export QUILT_PRIMARY_PACKAGE_DIR='/storage/quilt_modules'
cd super_resolution
# default to 10 epochs (prefer ~500)
N_EPOCHS=$1
echo "Training for ${N_EPOCHS:=10} epochs\n"
# train
python main.py \
	--upscale_factor 3 \
	--batchSize 4 \
	--testBatchSize 100 \
	--nEpochs $N_EPOCHS \
	--lr 0.001 \
	--cuda
