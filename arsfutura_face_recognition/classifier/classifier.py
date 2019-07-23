import os
import pickle
from . import FaceClassifier


class FaceClassifierImpl(FaceClassifier):
    def __init__(self):
        self.le, self.model = pickle.load(open(os.path.join(os.path.dirname(__file__), '../../models/model.pkl'), 'rb'))

    def predict(self, face_embedding):
        return self.le.inverse_transform(self.model.predict(face_embedding))
