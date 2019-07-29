from .classifier import FaceClassifierImpl


def classifier_factory():
    return FaceClassifierImpl()
