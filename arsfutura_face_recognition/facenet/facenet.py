from . import FaceNet
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1


class FaceNetImpl(FaceNet):
    def __init__(self):
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval()

    def forward(self, img):
        return self.facenet(img)
