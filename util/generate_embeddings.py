import argparse
import os
import joblib
import numpy as np
import torch
from torchvision import datasets
from face_recognition import FaceFeaturesExtractor
from training.train import dataset_to_embeddings


def parse_args():
    parser = argparse.ArgumentParser(
        "Script for generating face embeddings. Output of this script is 'embeddings.txt' which contains embeddings "
        "for all input images, 'labels.txt' which contains label for every embedding and 'class_to_idx.pkl' which "
        "is serializes dictionary which maps classes to its index.")
    parser.add_argument('--input-folder', required=True,
                        help='Root folder where images are. This folder contains sub-folders for each class.')
    parser.add_argument('--output-folder', required=True,
                        help='Output folder where image embeddings and labels will be saved.')
    return parser.parse_args()


def normalise_string(string):
    return string.lower().replace(' ', '_')


def normalise_dict_keys(dictionary):
    new_dict = dict()
    for key in dictionary.keys():
        new_dict[normalise_string(key)] = dictionary[key]
    return new_dict


def main():
    torch.set_grad_enabled(False)
    args = parse_args()

    features_extractor = FaceFeaturesExtractor()
    dataset = datasets.ImageFolder(args.input_folder)
    embeddings, labels = dataset_to_embeddings(dataset, features_extractor)

    dataset.class_to_idx = normalise_dict_keys(dataset.class_to_idx)
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    labels = list(map(lambda idx: idx_to_class[idx], labels))

    np.savetxt(args.output_folder + os.path.sep + 'embeddings.txt', embeddings)
    np.savetxt(args.output_folder + os.path.sep + 'labels.txt', np.array(labels, dtype=np.str).reshape(-1, 1), fmt="%s")
    joblib.dump(dataset.class_to_idx, args.output_folder + os.path.sep + 'class_to_idx.pkl')


if __name__ == '__main__':
    main()
