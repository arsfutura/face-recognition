import torch
from .aligner.factory import aligner_factory
from .facenet.factory import facenet_factory
from .classifier.factory import classifier_factory
from facenet_pytorch.models.utils.detect_face import extract_face
from collections import namedtuple

Face = namedtuple('Face', 'bb identity probability')


class BoundingBox:
    def __init__(self, left, top, right, bottom):
        self._left = left
        self._top = top
        self._right = right
        self._bottom = bottom

    def left(self):
        return self._left

    def top(self):
        return self._top

    def right(self):
        return self._right

    def bottom(self):
        return self._bottom


def face_recogniser_factory():
    return FaceRecogniser(
        aligner=aligner_factory(),
        facenet=facenet_factory(),
        classifier=classifier_factory()
    )


class FaceRecogniser:
    def __init__(self, aligner, facenet, classifier):
        self.aligner = aligner
        self.facenet = facenet
        self.classifier = classifier

    def recognise_faces(self, img):
        bbs, _ = self.aligner(img)
        if bbs is None:
            # if no face is detected
            return None

        faces = torch.stack([extract_face(img, bb) for bb in bbs])
        embeddings = self.facenet(faces).detach().numpy()
        people = self.classifier(embeddings)

        return [Face(BoundingBox(left=bb[0], top=bb[1], right=bb[2], bottom=bb[3]), person, 100)
                for bb, person in zip(bbs, people)]

    def __call__(self, *args, **kwargs):
        return self.recognise_faces(*args, **kwargs)
