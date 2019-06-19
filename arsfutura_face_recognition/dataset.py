import numpy as np


def load_data():
    return np.loadtxt('../embeddings.txt'), np.loadtxt('../labels.txt', dtype=np.str)


if __name__ == '__main__':
    print(load_data())
