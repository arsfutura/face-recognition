FROM python:3.7-slim-buster

MAINTAINER Luka Dulčić "culuma@arsfutura.co"

RUN mkdir -p /app && \
    apt-get update -y && \
    apt-get install -y build-essential python3-dev libsm6 libxext6 libxrender-dev libglib2.0-0

WORKDIR /app

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

RUN pip3 install -r requirements.txt

COPY face_recognition /app/face_recognition
COPY model /app/model
COPY api /app/api
COPY tasks/run_prod_server.sh /app/run_prod_server.sh
RUN chmod +x run_prod_server.sh

CMD [ "./run_prod_server.sh" ]
