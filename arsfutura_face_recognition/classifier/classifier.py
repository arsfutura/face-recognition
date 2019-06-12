from . import FaceClassifier
import pickle


class FaceClassifierImpl(FaceClassifier):
    def __init__(self, model_path):
        self.model = pickle.load(model_path)

    def predict(self, face_embedding):
        return self.model(face_embedding)
