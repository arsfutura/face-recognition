#!/usr/bin/env bash

python -m bin.exif_orientation_normalize --images-path $1
echo "===================================="
echo "Exif orientation normalization done."
echo "===================================="

python -m bin.align-mtcnn --input-folder $1 --output-folder $2
echo "===================================="
echo "Cropping faces (aligning) done."
echo "===================================="

python -m bin.embeddings --input-folder $1 --output-folder $3
echo "===================================="
echo "Generating face embeddings done."
echo "===================================="