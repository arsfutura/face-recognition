#!/usr/bin/env python

from arsfutura_face_recognition.face_recogniser import face_recogniser_factory
import argparse
import base64
import json
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        'Script for recognising faces on picture. Output of this script is json with list of people on picture and '
        'base64 encoded picture which has bounding boxes of people.')
    image_group = parser.add_mutually_exclusive_group(required=True)
    image_group.add_argument('--image-path', help='Path to image file.')
    image_group.add_argument('--image-bs64', help='Base64 representation of image.')
    parser.add_argument('--classifier-path', required=True, help='Path to serialized classifier.')
    return parser.parse_args()


def base64_to_img(bs64_img):
    decoded = base64.b64decode(bs64_img)
    return cv2.imdecode(np.frombuffer(decoded, dtype=np.uint8), flags=cv2.IMREAD_COLOR)


def img_to_base64(img):
    ret, buff = cv2.imencode('.png', img)
    return base64.b64encode(buff)


def load_image(args):
    if args.image_path:
        return cv2.imread(args.image_path)
    if args.image_bs64:
        return base64_to_img(args.image_bs64)


def draw_bb_on_img(faces, img):
    for face in faces:
        cv2.rectangle(img, (int(face.bb.left()), int(face.bb.top())), (int(face.bb.right()), int(face.bb.bottom())),
                      (0, 255, 0), 2)
        cv2.putText(img, "%s %.2f%%" % (face.identity, face.probability),
                    (int(face.bb.left()), int(face.bb.bottom()) + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)


def main():
    args = parse_args()
    img = load_image(args)
    faces = face_recogniser_factory(args)(img)
    draw_bb_on_img(faces, img)
    print(json.dumps(
        {
            'people': list(map(lambda f: f.identity, faces)),
            'img': str(img_to_base64(img), encoding='ascii')
        }
    ))


if __name__ == '__main__':
    main()
