import cv2
from PIL import Image
from .aligner.factory import aligner_factory
from .facenet.factory import facenet_factory
from .classifier.factory import classifier_factory
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


def face_recogniser_factory(args):
    return FaceRecogniser(
        aligner=aligner_factory(args),
        facenet=facenet_factory(args),
        classifier=classifier_factory(args)
    )


class FaceRecogniser:
    def __init__(self, aligner, facenet, classifier):
        self.aligner = aligner
        self.facenet = facenet
        self.classifier = classifier

    def recognise_faces(self, img):
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        aligned_img, bb = self.aligner(img_pil)
        if aligned_img is None:
            # if no face is detected
            return None

        embedding = self.facenet(aligned_img.unsqueeze(0)).detach().numpy()
        person = self.classifier(embedding)

        return [Face(BoundingBox(left=bb[0], top=bb[1], right=bb[2], bottom=bb[3]), person, 100)]

    def __call__(self, *args, **kwargs):
        return self.recognise_faces(*args, **kwargs)
