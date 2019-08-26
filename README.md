This repository provides a framework for creating and using Face Recognition system. 

# Installation
Make sure you have [Python 3](https://realpython.com/installing-python/) and 
[`pip`](https://www.makeuseof.com/tag/install-pip-for-python/) installed.

Install dependencies
```
pip install -r requirements.txt
```

# Train Face Recognition system
In order to train Face Recognition system user needs to provide images of people in a folder with following structure: 

```
- images 
    - person1
        - person1_1.png
        - person1_2.png
        ...
        - person1_n.png
    - person2
    ...
    - personN
    ...
```

Every directory contains images of one person (the more images the better), every image must contain only one face of a 
person.

After preparing images run following command to train Face Recognition system:
```
./tasks/train.sh path/to/folder/with/images
``` 
Previous command will generate `model/face_recogniser.pkl` which represents trained Face Recognition system.

# Using Face Recognition

## Scripts
TODO video, face recogniser

## Face Recognition API
You can use trained Face Recognition system as REST API, `api` folder contains simple 
[Flask](https://palletsprojects.com/p/flask/) API which provides frontend for Face Recognition system.

Run server using following command:
```
tasks/run_server.sh
```

Server is running on port `5000`.
Swagger API docs are available upon running server on `<root-url>:5000/docs`       

### Docker
Easiest way to run Face Recognition API is through Docker container.

Build image
```
docker build -t face-recognition-api:latest -f api/Dockerfile .
```

Run server
```
docker run --name face-recognition-api -d -p 5000:5000 face-recognition-api
```

> WARNING Face Recognition API memory usage depends on image sizes it's processing, if you plan to process 
high-resolution images Face Recognition API container will need at least 3GB of RAM (maybe more) to run, if you are 
running Docker containers with limited amount of RAM you could encounter OOM. 
