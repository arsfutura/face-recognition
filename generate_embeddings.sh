#!/usr/bin/env bash

python -m bin.exif_orientation_normalize --images-path $1
echo -e "\n===================================="
echo "Exif orientation normalization done."
echo "===================================="

python -m bin.align-mtcnn --input-folder $1 --output-folder $2
echo -e "\n===================================="
echo "Cropping faces (aligning) done."
echo "===================================="

python -m bin.generate_embeddings --input-folder $2 --output-folder $3
echo -e "\n===================================="
echo "Generating face embeddings done."
echo "===================================="