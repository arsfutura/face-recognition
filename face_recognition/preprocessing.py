from PIL import ImageOps


class ExifOrientationNormalize(object):
    """
    Normalizes rotation of the image based on exif orientation info (if exists.)
    """

    def __call__(self, img):
        return ImageOps.exif_transpose(img)


class FixedImageStandardization(object):
    """
    Normalizes image with fixed mean and stddev.
    """

    def __init__(self, mean=127.5, stddev=128):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, img):
        return (img - self.mean) / self.stddev
