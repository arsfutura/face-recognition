from facenet import MTCNN
from . import Aligner


class MTCNNAligner(Aligner):
    def __init__(self):
        self.mtcnn = MTCNN(keep_all=True)

    def align(self, img):
        return self.mtcnn(img)
