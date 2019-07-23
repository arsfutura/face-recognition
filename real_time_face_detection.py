#!/usr/bin/env python

import cv2
from PIL import Image
from .arsfutura_face_recognition import face_recogniser_factory


def main():
    cap = cv2.VideoCapture(0)
    face_recogniser = face_recogniser_factory()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        faces = face_recogniser(Image.fromarray(frame))
        if faces is not None:
            for face in faces:
                cv2.rectangle(frame, (int(face.bb.left()), int(face.bb.top())),
                              (int(face.bb.right()), int(face.bb.bottom())),
                              (0, 255, 0), 2)
                cv2.putText(frame, "%s %.2f%%" % (face.identity, face.probability),
                            (int(face.bb.left()), int(face.bb.bottom()) + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                            (255, 255, 255), 1)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the captureq
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
