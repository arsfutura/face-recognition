from .mtcnn import MTCNNAligner


def aligner_factory(args):
    return MTCNNAligner()
