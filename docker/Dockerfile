# FROM ubuntu:23.10
FROM python:3.10
# copy the whole code
COPY . /digits/

# RUN apt-get update
# RUN apt-get install -y python3 python3-pip
RUN pip3 install -r /digits/requirements.txt
# copy the requirements.txt
# COPY requirements.txt /requirements.tx
# need python
# no need fr conda or venv
# requirements installation

WORKDIR /digits
ENV FLASK_APP=api/__pycache__/quiz3

CMD ["flask","run","--host=0.0.0.0"]

# CMD ["pytest"]