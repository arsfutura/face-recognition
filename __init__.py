from .face_recognition import face_recogniser_factory

face_recognizer = face_recogniser_factory(include_predictions=True)


def recognise_faces(img):
    return face_recognizer(img)
