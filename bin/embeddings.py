import argparse
import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1


def parse_args():
    parser = argparse.ArgumentParser(
        "Script for generating face embeddings. Output of this script is 'embeddings.txt' which contains embeddings "
        "for all input images and 'labels.txt' which contains label for every embedding.")
    parser.add_argument('--input-folder', required=True,
                        help='Root folder where *aligned* images are. This folder contains sub-folders for each class.')
    parser.add_argument('--output-folder', required=True,
                        help='Output folder where image embeddings and labels will be saved.')
    return parser.parse_args()


def main():
    torch.set_grad_enabled(False)
    args = parse_args()

    aligned_images = datasets.ImageFolder(args.input_folder, transform=transforms.ToTensor())
    aligned_images.idx_to_class = {v: k for k, v in aligned_images.class_to_idx.items()}
    data_loader = DataLoader(aligned_images, batch_size=32)
    facenet = InceptionResnetV1(pretrained='vggface2').eval()

    embeddings = None
    labels = []
    for x, y in data_loader:
        embeddings = facenet(x) if embeddings is None else torch.cat([embeddings, facenet(x)], dim=0)
        labels += map(lambda idx: aligned_images.idx_to_class[idx], y.detach().numpy().tolist())
    np.savetxt(args.output_folder + os.path.sep + 'embeddings.txt', embeddings.detach().numpy())
    np.savetxt(args.output_folder + os.path.sep + 'labels.txt', np.array(labels, dtype=np.str).reshape(-1, 1), fmt="%s")


if __name__ == '__main__':
    main()
