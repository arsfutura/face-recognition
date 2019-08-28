FROM ubuntu:18.04

MAINTAINER Luka Dulčić "culuma@arsfutura.co"

RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev libsm6 libxext6 libxrender-dev

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip3 install -r requirements.txt

COPY face_recognition /app/face_recognition
COPY model /app/model
COPY api /app/api

CMD [ "python3", "-m", "api.app" ]