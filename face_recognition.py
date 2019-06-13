from arsfutura_face_recognition.face_recogniser import face_recogniser_factory
import argparse
import base64
import json
import cv2
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser('Script for recognising faces on picture.')
    parser.add_argument('--image', required=True, help='Base64 representation of image.')
    parser.add_argument('--classifier-path', required=True, help='Path to serialized classifier.')
    return parser.parse_args()


def base64_to_img(bs64_img):
    decoded = base64.b64decode(bs64_img)
    return cv2.imdecode(np.frombuffer(decoded, dtype=np.uint8), flags=cv2.IMREAD_COLOR)


def img_to_base64(img):
    ret, buff = cv2.imencode('.png', img)
    return base64.b64encode(buff)


def draw_bb_on_img(faces, img):
    for face in faces:
        cv2.rectangle(img, (face.bb.left(), face.bb.top()), (face.bb.right(), face.bb.bottom()), (0, 255, 0), 2)
        cv2.putText(img, "%s %.2f%%" % (face.identity, face.probability),
                    (face.bb.left(), face.bb.bottom() + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)


def main():
    args = parse_args()
    #img = Image.fromarray(base64_to_img(args.image))
    img = Image.open('1.jpg')
    faces = face_recogniser_factory(args)(img)
    draw_bb_on_img(faces, img)
    print(json.dumps(
        {
            'people': faces,
            'img': img_to_base64(img)
        }
    ))


if __name__ == '__main__':
    main()
