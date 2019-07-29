from .mtcnn import MTCNNAligner


def aligner_factory():
    return MTCNNAligner()
