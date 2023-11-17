#!/bin/bash

IMAGE="digits:v1"



docker build -t $IMAGE -f docker/Dockerfile .
docker volume create mlmodels
docker run -v mlmodels:/digits/Models $IMAGE

