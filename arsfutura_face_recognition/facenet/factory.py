from .facenet import FaceNetImpl


def facenet_factory():
    return FaceNetImpl()
