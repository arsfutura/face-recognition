#!/usr/bin/env python

import joblib
import cv2
import numpy as np
from PIL import Image
from face_recognition import preprocessing
from .util import draw_bb_on_img
from .constants import MODEL_PATH


def main():
    cap = cv2.VideoCapture(0)
    face_recogniser = joblib.load(MODEL_PATH)
    preprocess = preprocessing.ExifOrientationNormalize()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        img = Image.fromarray(frame)
        faces = face_recogniser(preprocess(img))
        if faces is not None:
            draw_bb_on_img(faces, img)

        # Display the resulting frame
        cv2.imshow('video', np.array(img))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the captureq
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
