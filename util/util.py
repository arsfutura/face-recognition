import os
import numpy as np
import joblib
from datetime import datetime
from pathlib import Path
from PIL import Image
from torchvision import transforms
from face_recognition import preprocessing

EMBEDDINGS_FILE = 'embeddings.tsv'
LABELS_FILE = 'labels.tsv'
PATHS_FILE = 'paths.tsv'
CLASS_TO_IDX_FILE = 'class_to_idx.pkl'


def save_embeddings_data(folder, embeddings, labels, paths, class_to_idx):
    root_dir = Path('{}{}{}-{}'.format(folder, os.path.sep, 'embeddings', datetime.now()))
    root_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(root_dir.joinpath(EMBEDDINGS_FILE).absolute(), embeddings, delimiter='\t')
    np.savetxt(root_dir.joinpath(LABELS_FILE).absolute(), np.array(labels, dtype=np.str).reshape(-1, 1), fmt="%s",
               delimiter='\t')
    np.savetxt(root_dir.joinpath(PATHS_FILE).absolute(), np.array(paths, dtype=np.str).reshape(-1, 1), fmt="%s",
               delimiter='\t')
    joblib.dump(class_to_idx, root_dir.joinpath(CLASS_TO_IDX_FILE).absolute())


def load_embeddings_data(folder):
    root_dir = Path(folder)
    embeddings = np.loadtxt(root_dir.joinpath(EMBEDDINGS_FILE), delimiter='\t')
    labels = np.loadtxt(root_dir.joinpath(LABELS_FILE), dtype='str', delimiter='\t')
    paths = np.loadtxt(root_dir.joinpath(PATHS_FILE), dtype='str', delimiter='\t')
    class_to_idx = joblib.load(root_dir.joinpath(CLASS_TO_IDX_FILE).absolute())
    return embeddings, labels, paths, class_to_idx


def dataset_to_embeddings(dataset, features_extractor, cache=None):
    transform = transforms.Compose([
        preprocessing.ExifOrientationNormalize(),
        transforms.Resize(1024)
    ])

    embeddings = []
    labels = []
    paths = []
    for img_path, label in dataset.samples:
        print(img_path, end='')
        if cache is not None and img_path in cache:
            embedding = cache[img_path]
            print(' Found in cache.')
        else:
            _, embedding = features_extractor(transform(Image.open(img_path).convert('RGB')))
            if embedding is None:
                print("\nCould not find face on {}".format(img_path))
                continue
            if embedding.shape[0] > 1:
                print("\nMultiple faces detected for {}, taking one with highest probability".format(img_path))
                embedding = embedding[0, :]
            embedding = embedding.flatten()
            print()

        embeddings.append(embedding)
        labels.append(label)
        paths.append(img_path)

    return np.stack(embeddings), labels, paths
