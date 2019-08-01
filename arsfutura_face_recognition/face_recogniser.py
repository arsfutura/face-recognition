import torch
import pickle
import os
from .aligner.factory import aligner_factory
from .facenet.factory import facenet_factory
from facenet_pytorch.models.utils.detect_face import extract_face
from collections import namedtuple

Prediction = namedtuple('Prediction', 'id name confidence')
Face = namedtuple('Face', 'top_prediction bb all_predictions')
BoundingBox = namedtuple('BoundingBox', 'left top right bottom')


def face_recogniser_factory(include_predictions=False):
    return FaceRecogniser(
        aligner=aligner_factory(),
        facenet=facenet_factory(),
        include_predictions=include_predictions
    )


def top_prediction(le, probs):
    top_label = probs.argmax()
    return Prediction(id=top_label, name=le.classes_[top_label], confidence=probs[top_label])


def to_predictions(le, probs):
    return [Prediction(id=i, name=le.classes_[i], confidence=prob) for i, prob in enumerate(probs)]


class FaceRecogniser:
    def __init__(self, aligner, facenet, include_predictions):
        self.aligner = aligner
        self.facenet = facenet
        self.le, self.classifier = pickle.load(
            open(os.path.join(os.path.dirname(__file__), '../models/model.pkl'), 'rb'))
        self.include_predictions = include_predictions

    def recognise_faces(self, img):
        bbs, _ = self.aligner(img)
        if bbs is None:
            # if no face is detected
            return []

        faces = torch.stack([extract_face(img, bb) for bb in bbs])
        embeddings = self.facenet(faces).detach().numpy()
        predictions = self.classifier.predict_proba(embeddings)

        return [
            Face(
                top_prediction=top_prediction(self.le, probs),
                bb=BoundingBox(left=bb[0], top=bb[1], right=bb[2], bottom=bb[3]),
                all_predictions=None if not self.include_predictions else to_predictions(self.le, probs)
            )
            for bb, probs in zip(bbs, predictions)
        ]

    def __call__(self, *args, **kwargs):
        return self.recognise_faces(*args, **kwargs)
