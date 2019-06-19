from . import FaceNet
from facenet import InceptionResnetV1


class FaceNetImpl(FaceNet):
    def __init__(self):
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval()

    def forward(self, img):
        return self.facenet(img)
