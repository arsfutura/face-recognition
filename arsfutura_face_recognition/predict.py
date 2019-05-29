import openface
import cv2
import pickle
import numpy as np
from collections import namedtuple
from constants import MODEL_PATH

Face = namedtuple('Face', 'bb identity probability')

align = openface.AlignDlib('../models/shape_predictor_68_face_landmarks.dat')
net = openface.TorchNeuralNet('../models/nn4.small2.v1.t7', imgDim=96, cuda=False)
le, model = pickle.load(open(MODEL_PATH, 'rb'))


def faces_embeddings_and_bbs(img):
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bb = align.getAllFaceBoundingBoxes(rgb_img)

    aligned_faces = []
    for box in bb:
        aligned_faces.append(
            align.align(
                96,
                rgb_img,
                box,
                landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE))

    embeddings = np.array([net.forward(alignedFace) for alignedFace in aligned_faces])

    return embeddings, bb


def predict(img):
    embeddings, bb = faces_embeddings_and_bbs(img)
    if embeddings.size == 0 or not bb:
        return None
    probs = model.predict_proba(embeddings)
    predicted = probs.argmax(axis=1)
    identities = le.inverse_transform(predicted)
    probs = probs[np.arange(len(predicted)), predicted]
    return [Face(box, identity, prob * 100) for identity, box, prob in zip(identities, bb, probs)]


if __name__ == '__main__':
    predict('')
