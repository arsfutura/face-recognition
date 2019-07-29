from .arsfutura_face_recognition import face_recogniser_factory

face_recognizer = face_recogniser_factory()


def recognise_faces(img):
    return face_recognizer(img)
