from facenet_pytorch import MTCNN
from . import Aligner
from torchvision import transforms
from .. import preprocessing


class MTCNNAligner(Aligner):
    def __init__(self):
        self.mtcnn = MTCNN(keep_all=True)
        self.preprocess = transforms.Compose([
            preprocessing.ExifOrientationNormalize()
        ])

    def align(self, img):
        return self.mtcnn.detect(self.preprocess(img))
