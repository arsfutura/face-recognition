import os
import numpy as np
import joblib
from datetime import datetime
from pathlib import Path


def save_embeddings_data(folder, embeddings, labels, paths, class_to_idx):
    root_dir = Path('{}{}{}-{}'.format(folder, os.path.sep, 'embeddings', datetime.now()))
    root_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(root_dir.joinpath('embeddings.tsv').absolute(), embeddings, delimiter='\t')
    np.savetxt(root_dir.joinpath('labels.tsv').absolute(), np.array(labels, dtype=np.str).reshape(-1, 1), fmt="%s",
               delimiter='\t')
    np.savetxt(root_dir.joinpath('paths.tsv').absolute(), np.array(paths, dtype=np.str).reshape(-1, 1), fmt="%s",
               delimiter='\t')
    joblib.dump(class_to_idx, root_dir.joinpath('class_to_idx.pkl').absolute())
