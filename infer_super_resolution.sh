#!/usr/bin/bash
# USAGE: bash infer_super_resolution.sh EPOCH TRAIN_IMAGE_ID
# Entrypoint for Paperspace Job

# read/write Quilt packages from/to persistent storage
export QUILT_PRIMARY_PACKAGE_DIR='/storage/quilt_packages'
# working directory
cd super_resolution
# requirements
pip install -r requirements.txt

quilt export akarve/BSDS300/images/test/n$2 .

python super_resolve.py \
	--cuda \
	--input_image images/test/$2.jpg \
	--model /storage/models/super_resolution/model_epoch_$1.pth \
	--output_filename /artifacts/super-$2.png
