from . import FaceNet
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from .. import preprocessing


class FaceNetImpl(FaceNet):
    def __init__(self):
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval()
        self.preprocess = transforms.Compose([
            preprocessing.Whitening()
        ])

    def forward(self, img):
        return self.facenet(self.preprocess(img))
