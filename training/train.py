import os
import argparse
import joblib
from torchvision import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from face_recognition import FaceFeaturesExtractor, FaceRecogniser
from util.util import load_embeddings_data, dataset_to_embeddings

MODEL_DIR_PATH = 'model'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for training Face Recognition model. You can either give path to dataset or provide path '
                    'to pre-generated embeddings (--embeddings-path), labels and class_to_idx. You can pre-generate '
                    'this with util/generate_embeddings.py script.')
    parser.add_argument('-d', '--dataset-path', help='Path to folder with images.')
    parser.add_argument('-e', '--embeddings-path', help='Path to folder with embeddings data.')
    parser.add_argument('--grid-search', action='store_true',
                        help='If this option is enabled, grid search will be performed to estimate C parameter of '
                             'Logistic Regression classifier. In order to use this option you have to have at least '
                             '3 examples of every class in your dataset. It is recommended to enable this option.')
    return parser.parse_args()


def load_data(args, features_extractor):
    if args.embeddings_path:
        embeddings, labels, _, class_to_idx = load_embeddings_data(args.embeddings_path)
        return embeddings, labels.tolist(), class_to_idx

    dataset = datasets.ImageFolder(args.dataset_path)
    embeddings, labels, _ = dataset_to_embeddings(dataset, features_extractor)
    return embeddings, labels, dataset.class_to_idx


def train(args, embeddings, labels):
    softmax = LogisticRegression(solver='lbfgs', multi_class='multinomial', C=10, tol=1E-8, max_iter=100000)
    if args.grid_search:
        clf = GridSearchCV(
            estimator=softmax,
            param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
            cv=3,
            n_jobs=-1,
            verbose=1
        )
    else:
        clf = softmax
    clf.fit(embeddings, labels)

    return clf.best_estimator_ if args.grid_search else clf


def main():
    args = parse_args()

    features_extractor = FaceFeaturesExtractor()
    embeddings, labels, class_to_idx = load_data(args, features_extractor)
    clf = train(args, embeddings, labels)

    idx_to_class = {v: k for k, v in class_to_idx.items()}

    target_names = list(map(lambda i: i[1], sorted(idx_to_class.items(), key=lambda i: i[0])))
    print(metrics.classification_report(labels, clf.predict(embeddings), target_names=target_names))

    if not os.path.isdir(MODEL_DIR_PATH):
        os.mkdir(MODEL_DIR_PATH)
    model_path = os.path.join('model', 'face_recogniser.pkl')
    joblib.dump(FaceRecogniser(features_extractor, clf, idx_to_class), model_path)


if __name__ == '__main__':
    main()
