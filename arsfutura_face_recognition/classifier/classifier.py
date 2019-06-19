from . import FaceClassifier
import pickle


class FaceClassifierImpl(FaceClassifier):
    def __init__(self, model_path):
        self.le, self.model = pickle.load(open(model_path, 'rb'))

    def predict(self, face_embedding):
        return self.le.inverse_transform(self.model.predict(face_embedding))
