from collections import OrderedDict

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
    X, y = load_data()
    cls = METHODS['tsne']
    method = cls(n_components=2)
    transformed = method.fit_transform(X)

    y = np.array(y)
    plt.figure(figsize=(10, 12))
    colors = cm.rainbow(np.linspace(0, 1, len(y)))
    for label, color in zip(y, colors):
        points = transformed[y == label, :]
        plt.scatter(points[:, 0], points[:, 1], c=color, label=label, s=200, alpha=0.5)
        plt.annotate(label, (points[0, 0], points[0, 1]), fontsize=15)

    #plt.grid()
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.show()


if __name__ == '__main__':
    main()
