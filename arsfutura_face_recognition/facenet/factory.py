from .facenet import FaceNetImpl


def facenet_factory(args):
    return FaceNetImpl()
