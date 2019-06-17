### `face_recognition.py`
```
usage: Script for recognising faces on picture. Output of this script is json with list of people on picture and base64 encoded picture which has bounding boxes of people.
       [-h] (--image-path IMAGE_PATH | --image-bs64 IMAGE_BS64)
       --classifier-path CLASSIFIER_PATH

optional arguments:
  -h, --help            show this help message and exit
  --image-path IMAGE_PATH
                        Path to image file.
  --image-bs64 IMAGE_BS64
                        Base64 representation of image.
  --classifier-path CLASSIFIER_PATH
                        Path to serialized classifier.
```

#

### `real_time_face_detection.py`
```
usage: Script for real-time face recognition. [-h]
                                              [--classifier-path CLASSIFIER_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --classifier-path CLASSIFIER_PATH
                        Path to serialized classifier.
```                   