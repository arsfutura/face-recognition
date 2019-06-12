from abc import ABC, abstractmethod


class Aligner(ABC):
    @abstractmethod
    def align(self, img):
        pass

    def __call__(self, img, *args, **kwargs):
        self.align(img)
