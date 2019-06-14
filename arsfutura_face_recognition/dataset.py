import pandas as pd
import numpy as np
from constants import DATA_PATH

EMBEDDINGS_PATH = DATA_PATH + 'reps.csv'
LABELS_PATH = DATA_PATH + 'labels.csv'


def load_data():
    # embeddings = pd.read_csv(EMBEDDINGS_PATH, header=None).values
    # labels = list(map(lambda label: label.split('/')[-2].upper(), pd.read_csv(LABELS_PATH, header=None).values[:, 1].tolist()))
    # return embeddings, labels
    return np.loadtxt('embeddings.txt'), np.loadtxt('labels.txt', dtype=np.str)


if __name__ == '__main__':
    print(load_data())
