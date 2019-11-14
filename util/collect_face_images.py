import os
import cv2
import argparse


def main(directory, name, test):
    cap = cv2.VideoCapture(0)

    i = 0
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        # Display the resulting frame
        cv2.imshow(name, frame)
        if not test and i != 0 and i % 10 == 0:
            cv2.imwrite("{}/{}{}.png".format(directory, name, int(i / 10)), frame)
        i += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the captureq
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--person', required=True)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    directory = 'images/{}'.format(args.person)
    if not args.test and not os.path.exists(directory):
        os.mkdir(directory)
    try:
        main(directory, args.person, args.test)
    except KeyboardInterrupt:
        print("Photo session done for {} :)".format(args.person))
