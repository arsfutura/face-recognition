from .aligner.factory import aligner_factory
from .facenet.factory import facenet_factory
from .classifier.factory import classifier_factory


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
        aligned = self.aligner(img) # TODO no face
        embedding = self.facenet(aligned.unsqueeze(0)).detach().numpy()
        return self.classifier(embedding)

    def __call__(self, *args, **kwargs):
        return self.recognise_faces(*args, **kwargs)
