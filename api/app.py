import io
import joblib
from PIL import Image
from flask import Flask
from flask_restplus import Api, Resource, fields, abort, inputs
from werkzeug.datastructures import FileStorage
from face_recognition import preprocessing

face_recogniser = joblib.load('model/face_recogniser.pkl')
preprocess = preprocessing.ExifOrientationNormalize()

IMAGE_KEY = 'image'
INCLUDE_PREDICTIONS_KEY = 'include_predictions'
app = Flask(__name__)
api = Api(app, version='0.1.0', title='Face Recognition API', doc='/docs')

parser = api.parser()
parser.add_argument(IMAGE_KEY, type=FileStorage, location='files', required=True,
                    help='Image on which face recognition will be run.')
parser.add_argument(INCLUDE_PREDICTIONS_KEY, type=inputs.boolean, default=False,
                    help='Whether to include all predictions in response.')

bounding_box = api.model('BoundingBox', {
    'left': fields.Float,
    'top': fields.Float,
    'right': fields.Float,
    'bottom': fields.Float,
})

prediction = api.model('Prediction', {
    'label': fields.String,
    'confidence': fields.Float
})

face_model = api.model('Face', {
    'top_prediction': fields.Nested(prediction),
    'bounding_box': fields.Nested(bounding_box),
    'all_predictions': fields.List(fields.Nested(prediction))
})

response_model = api.model('Response', {
    'faces': fields.List(fields.Nested(face_model))
})

error_model = api.model('ErrorResponse', {
    'message': fields.String
})


@api.route('/face-recognition')
class FaceRecognition(Resource):
    @api.expect(parser, validate=True)
    @api.marshal_with(response_model)
    @api.response(200, 'Success')
    @api.response(400, 'No image file in request.', error_model)
    def post(self):
        args = parser.parse_args()
        if IMAGE_KEY not in args:
            abort(400, "Image field '{}' doesn't exist in request!".format(IMAGE_KEY))

        img = Image.open(io.BytesIO(args[IMAGE_KEY].read()))
        img = preprocess(img)
        # convert image to RGB (stripping alpha channel if exists)
        img = img.convert('RGB')
        faces = face_recogniser(img)
        return \
            {
                'faces': [
                    {
                        'top_prediction': face.top_prediction._asdict(),
                        'bounding_box': face.bb._asdict(),
                        'all_predictions': [p._asdict() for p in face.all_predictions] if
                        args[INCLUDE_PREDICTIONS_KEY] else None
                    }
                    for face in faces
                ]
            }


if __name__ == '__main__':
    app.run(host='0.0.0.0')
