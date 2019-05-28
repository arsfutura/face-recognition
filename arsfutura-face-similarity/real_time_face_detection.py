from __future__ import print_function, division

import cv2
from predict import predict


def main():
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        faces = predict(frame)
        if faces is not None:
            for face in faces:
                cv2.rectangle(frame, (face.bb.left(), face.bb.top()), (face.bb.right(), face.bb.bottom()), (0, 255, 0), 2)
                cv2.putText(frame, "%s %.2f%%" % (face.identity, face.probability),
                            (face.bb.left(), face.bb.bottom() + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the captureq
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
