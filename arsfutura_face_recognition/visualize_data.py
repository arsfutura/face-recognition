from collections import OrderedDict
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from functools import partial
from dataset import load_data
import seaborn as sns
sns.set()


METHODS = {
    'pca': PCA,
    'tsne': partial(TSNE, perplexity=5, init='pca', learning_rate=200, verbose=1)
}


def main():
    X, labels = load_data()
    cls = METHODS['tsne']
    method = cls(n_components=2)
    transformed = method.fit_transform(X)

    y = set(labels)
    labels = np.array(labels)
    plt.figure(figsize=(10, 12))
    colors = cm.rainbow(np.linspace(0, 1, len(y)))
    for label, color in zip(y, colors):
        points = transformed[labels == label, :]
        plt.scatter(points[:, 0], points[:, 1], c=[color], label=label, s=200, alpha=0.5)
        for p1, p2 in random.sample(list(zip(points[:, 0], points[:, 1])), k=min(1, len(points))):
            plt.annotate(label, (p1, p2), fontsize=15)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.show()


if __name__ == '__main__':
    main()
