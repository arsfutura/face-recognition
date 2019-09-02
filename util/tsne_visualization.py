import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from sklearn.manifold import TSNE
import seaborn as sns

sns.set()


def parse_args():
    parser = argparse.ArgumentParser('Script for visualising face embedding vectors with TSNE in 2D.')
    parser.add_argument('-e', '--embeddings-path', required=True,
                        help='Path to file with embeddings. File must be numpy matrix in txt format.')
    parser.add_argument('-l', '--labels-path', required=True,
                        help='Path to file with labels. File must be numpy matrix in txt format.')
    return parser.parse_args()


def main():
    args = parse_args()
    X, labels = np.loadtxt(args.embeddings_path), np.loadtxt(args.labels_path, dtype=np.str)
    tsne = TSNE(n_components=2, n_iter=10000, perplexity=5, init='pca', learning_rate=200, verbose=1)
    transformed = tsne.fit_transform(X)

    y = set(labels)
    labels = np.array(labels)
    plt.figure(figsize=(20, 14))
    colors = cm.rainbow(np.linspace(0, 1, len(y)))
    for label, color in zip(y, colors):
        points = transformed[labels == label, :]
        plt.scatter(points[:, 0], points[:, 1], c=[color], label=label, s=200, alpha=0.5)
        for p1, p2 in random.sample(list(zip(points[:, 0], points[:, 1])), k=min(1, len(points))):
            plt.annotate(label, (p1, p2), fontsize=30)

    plt.savefig('tsne_visualization.png', transparent=True, bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == '__main__':
    main()
