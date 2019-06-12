from facenet_pytorch.models.mtcnn import MTCNN
from . import Aligner


class MTCNNAligner(Aligner):
    def __init__(self):
        self.mtcnn = MTCNN()

    def align(self, img):
        return self.mtcnn(img)
