import argparse
import torch
from torchvision import datasets
from face_recognition import FaceFeaturesExtractor
from training.train import dataset_to_embeddings
from .util import save_embeddings_data, load_embeddings_data


def parse_args():
    parser = argparse.ArgumentParser(
        "Script for generating face embeddings. Output of this script is 'embeddings.txt' which contains embeddings "
        "for all input images, 'labels.txt' which contains label for every embedding and 'class_to_idx.pkl' which "
        "is serializes dictionary which maps classes to its index.")
    parser.add_argument('--input-folder', required=True,
                        help='Root folder where images are. This folder contains sub-folders for each class.')
    parser.add_argument('--output-folder', required=True,
                        help='Output folder where image embeddings and labels will be saved.')
    parser.add_argument('--cache-folder',
                        help='Root folder where images are. This folder contains sub-folders for each class.')
    return parser.parse_args()


def normalise_string(string):
    return string.lower().replace(' ', '_')


def normalise_dict_keys(dictionary):
    new_dict = dict()
    for key in dictionary.keys():
        new_dict[normalise_string(key)] = dictionary[key]
    return new_dict


def load_cache_if_exists(cache_folder):
    if not cache_folder:
        return None

    embeddings, _, paths, _ = load_embeddings_data(cache_folder)
    assert embeddings.shape[0] == paths.shape[0], 'Embeddings and paths must have same size'
    return {paths[i]: embeddings[i] for i in range(embeddings.shape[0])}


def main():
    torch.set_grad_enabled(False)
    args = parse_args()

    cache = load_cache_if_exists(args.cache_folder)

    features_extractor = FaceFeaturesExtractor()
    dataset = datasets.ImageFolder(args.input_folder)
    embeddings, labels, paths = dataset_to_embeddings(dataset, features_extractor, cache)

    dataset.class_to_idx = normalise_dict_keys(dataset.class_to_idx)
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    labels = list(map(lambda idx: idx_to_class[idx], labels))

    save_embeddings_data(args.output_folder, embeddings, labels, paths, dataset.class_to_idx)


if __name__ == '__main__':
    main()
