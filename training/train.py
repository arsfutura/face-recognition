import os
import argparse
import joblib
import numpy as np
from PIL import Image
from torchvision import transforms, datasets
from sklearn.linear_model import LogisticRegression
from face_recognition import preprocessing, FaceFeaturesExtractor, FaceRecogniser


MODEL_DIR_PATH = 'model'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset-path', required=True, help='Path to folder with images.')
    return parser.parse_args()


def dataset_to_embeddings(dataset, features_extractor):
    transform = transforms.Compose([
            preprocessing.ExifOrientationNormalize(),
            transforms.Resize(1024)
        ])

    embeddings = []
    labels = []
    for img_path, label in dataset.samples:
        print(img_path)
        _, embedding = features_extractor(transform(Image.open(img_path).convert('RGB')))
        if embedding is None:
            print("Could not find face on {}".format(img_path))
            continue
        if embedding.shape[0] > 1:
            print("Multiple faces detected for {}, taking one with highest probability".format(img_path))
            embedding = embedding[0, :]
        embeddings.append(embedding.flatten())
        labels.append(label)

    return np.stack(embeddings), labels


def main():
    args = parse_args()

    features_extractor = FaceFeaturesExtractor()
    dataset = datasets.ImageFolder(args.dataset_path)
    embeddings, labels = dataset_to_embeddings(dataset, features_extractor)

    clf = LogisticRegression(C=10, solver='lbfgs', multi_class='multinomial')
    clf.fit(embeddings, labels)

    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    if not os.path.isdir(MODEL_DIR_PATH):
        os.mkdir(MODEL_DIR_PATH)
    model_path = os.path.join('model', 'face_recogniser.pkl')
    joblib.dump(FaceRecogniser(features_extractor, clf, idx_to_class), model_path)


if __name__ == '__main__':
    main()
