from abc import ABC, abstractmethod


class FaceClassifier(ABC):
    @abstractmethod
    def predict(self, face_embedding):
        pass

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)
