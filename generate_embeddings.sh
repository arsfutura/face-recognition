#!/usr/bin/env bash

python -m bin.align-mtcnn --input-folder data/images --output-folder data/aligned
echo -e "\n===================================="
echo "Cropping faces (aligning) done."
echo "===================================="

python -m bin.generate_embeddings --input-folder data/aligned --output-folder .
echo -e "\n===================================="
echo "Generating face embeddings done."
echo "===================================="