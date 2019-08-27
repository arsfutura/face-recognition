#!/usr/bin/env python

import os
import argparse
from torchvision import datasets, transforms
from facenet_pytorch.models.mtcnn import MTCNN
from face_recognition import preprocessing
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-folder', required=True,
                        help='Root folder where input images are. This folder contains sub-folders for each class.')
    parser.add_argument('--output-folder', required=True, help='Output folder where aligned images will be saved.')
    return parser.parse_args()


def create_dirs(root_dir, classes):
    if not os.path.isdir(root_dir):
        os.mkdir(root_dir)
    for clazz in classes:
        path = root_dir + os.path.sep + clazz
        if not os.path.isdir(path):
            os.mkdir(path)


def main():
    args = parse_args()
    trans = transforms.Compose([
        preprocessing.ExifOrientationNormalize(),
        transforms.Resize(1024)
    ])

    images = datasets.ImageFolder(root=args.input_folder)
    images.idx_to_class = {v: k for k, v in images.class_to_idx.items()}
    create_dirs(args.output_folder, images.classes)

    mtcnn = MTCNN(prewhiten=False)

    for idx, (path, y) in enumerate(images.imgs):
        print("Aligning {} {}/{} ".format(path, idx + 1, len(images)), end='')
        aligned_path = args.output_folder + os.path.sep + images.idx_to_class[y] + os.path.sep + os.path.basename(path)
        if not os.path.exists(aligned_path):
            img = mtcnn(img=trans(Image.open(path).convert('RGB')), save_path=aligned_path)
            print("No face found" if img is None else '')
        else:
            print('Already aligned')


if __name__ == '__main__':
    main()
