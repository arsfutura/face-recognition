# Face Recognition API
Simple [Flask](https://palletsprojects.com/p/flask/) API which provides frontend for 
[Ars Futura Face Recognition System](https://github.com/arsfutura/arsfutura-face-recognition).

Server is running on port `5000`.
Swagger API docs are available upon running server on `<root-url>:5000/docs`

## Installation
Make sure you have [Python 3](https://realpython.com/installing-python/) and 
[`pip`](https://www.makeuseof.com/tag/install-pip-for-python/) installed.

Install dependencies
```
pip install -r requirements.txt
```

## Run server 
```
python3 app.py
``` 

## Docker
Easiest way to run Face Recognition API is through Docker container.

Build image
```
docker build -t face-recognition-api:latest .
```

Run server
```
docker run --name face-recognition-api -d -p 5000:5000 face-recognition-api
```

> WARNING Face Recognition API needs at least 3GB of RAM (4GB ideally) to run, if you are running on resource 
constrained machines or Docker containers with less than 3GB of RAM you could encounter OOM. 
