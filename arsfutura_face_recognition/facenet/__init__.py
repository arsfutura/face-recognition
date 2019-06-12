from abc import ABC, abstractmethod


class FaceNet(ABC):
    @abstractmethod
    def forward(self, img):
        pass

    def __call__(self, *args, **kwargs):
        self.forward(*args, **kwargs)
