import os
from torchvision.datasets.folder import IMG_EXTENSIONS, has_file_allowed_extension, pil_loader
import joblib
import argparse
from inference.util import draw_bb_on_img
from inference.constants import MODEL_PATH
from face_recognition import preprocessing


def parse_args():
    parser = argparse.ArgumentParser(
        'Script for detecting and classifying faces on user-provided images. This script will process images from input'
        ' folder, draw bounding boxes and labels on images and save them to the output folder.')
    parser.add_argument('--input-folder', required=True, help='Folder where input images are.')
    parser.add_argument('--output-folder', required=True, help='Folder where tagged images will be saved')
    return parser.parse_args()


def recognise_faces(model, img):
    faces = model(img)
    if faces:
        draw_bb_on_img(faces, img)
    return faces, img


def main():
    args = parse_args()
    model = joblib.load(MODEL_PATH)
    preprocess = preprocessing.ExifOrientationNormalize()

    for fname in filter(lambda p: has_file_allowed_extension(p, IMG_EXTENSIONS), os.listdir(args.input_folder)):
        print('Processing {}...'.format(fname), end='', flush=True)
        img = preprocess(pil_loader(os.path.join(args.input_folder, fname)))
        faces, img = recognise_faces(model, img)
        if faces:
            img.save(os.path.join(args.output_folder, fname))
            print('Done')
        else:
            print('No faces, skipping')


if __name__ == '__main__':
    main()
