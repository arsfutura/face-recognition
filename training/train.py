import os
import argparse
import joblib
import numpy as np
from torchvision import transforms, datasets
from sklearn.linear_model import LogisticRegression
from face_recognition import preprocessing, FaceFeaturesExtractor, FaceRecogniser


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset-path', required=True, help='Path to folder with images.')
    parser.add_argument('-o', '--output-path', default='..',
                        help='Path where trained model and label encoder will be saved.')
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = datasets.ImageFolder(
        args.dataset_path,
        transform=transforms.Compose([
            preprocessing.ExifOrientationNormalize()
            # TODO resizing of image
        ]))

    features_extractor = FaceFeaturesExtractor()
    embeddings = np.stack([features_extractor(img)[1].flatten() for img, _ in dataset])

    clf = LogisticRegression(C=10)
    clf.fit(embeddings, dataset.targets)

    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    model_path = os.path.join(args.output_path, 'face_recogniser.pkl')
    joblib.dump(FaceRecogniser(features_extractor, clf, idx_to_class), model_path)


if __name__ == '__main__':
    main()
