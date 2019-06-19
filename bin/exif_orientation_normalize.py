#!/usr/bin/env python

import argparse
from torchvision.datasets import ImageFolder
from PIL import Image

# EXIF orientation info http://sylvana.net/jpegcrop/exif_orientation.html

exif_orientation_tag = 0x0112
exif_transpose_sequences = [  # Val  0th row  0th col
    [],  # 0    (reserved)
    [],  # 1   top      left
    [Image.FLIP_LEFT_RIGHT],  # 2   top      right
    [Image.ROTATE_180],  # 3   bottom   right
    [Image.FLIP_TOP_BOTTOM],  # 4   bottom   left
    [Image.FLIP_LEFT_RIGHT, Image.ROTATE_90],  # 5   left     top
    [Image.ROTATE_270],  # 6   right    top
    [Image.FLIP_TOP_BOTTOM, Image.ROTATE_90],  # 7   right    bottom
    [Image.ROTATE_90],  # 8   left     bottom
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script for normalizing images orientation based on exif orientation tag. "
                    "This script will search for exif orientation tag in image, if it exists script will change image "
                    "orientation to 1 (top, left side). These changes will be saved to image and exif info will be "
                    "erased from image. If image doesn't have exif info, this script will leave it unchanged")
    parser.add_argument('--images-path', required=True, help='Path to folder with images.')
    return parser.parse_args()


def main():
    args = parse_args()
    images = ImageFolder(args.images_path)

    for img_path, y in images.imgs:
        img = Image.open(img_path).convert('RGB')
        if 'parsed_exif' in img.info:
            orientation = img.info['parsed_exif'][exif_orientation_tag]
            transposes = exif_transpose_sequences[orientation]
            for trans in transposes:
                img = img.transpose(trans)
            print('Processing {} with orientation {}'.format(img_path, orientation))
            img.save(img_path)


if __name__ == '__main__':
    main()
