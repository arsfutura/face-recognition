from .classifier import FaceClassifierImpl


def classifier_factory(args):
    return FaceClassifierImpl(args.model_path)
