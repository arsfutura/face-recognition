#!/usr/bin/env python

import argparse
import joblib
import cv2
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-path', required=True, help='Path to face recogniser model.')
    return parser.parse_args()


def main():
    args = parse_args()
    cap = cv2.VideoCapture(0)
    face_recogniser = joblib.load(args.model_path)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        faces = face_recogniser(Image.fromarray(frame))
        if faces is not None:
            for face in faces:
                cv2.rectangle(frame, (int(face.bb.left), int(face.bb.top)), (int(face.bb.right), int(face.bb.bottom)),
                              (0, 255, 0), 2)
                cv2.putText(frame, "%s %.2f%%" % (face.top_prediction.name.upper(), face.top_prediction.confidence * 100),
                            (int(face.bb.left), int(face.bb.top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 4,
                            cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the captureq
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
