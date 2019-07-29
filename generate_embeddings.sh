#!/usr/bin/env bash

python -m bin.exif_orientation_normalize --images-path data/images
echo -e "\n===================================="
echo "Exif orientation normalization done."
echo "===================================="

python -m bin.align-mtcnn --input-folder data/images --output-folder data/aligned
echo -e "\n===================================="
echo "Cropping faces (aligning) done."
echo "===================================="

python -m bin.generate_embeddings --input-folder data/aligned --output-folder .
echo -e "\n===================================="
echo "Generating face embeddings done."
echo "===================================="