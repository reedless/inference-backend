# sudo docker build -t pytorch_cpu -f Dockerfile .

FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
  libopencv-dev \
  python3-pip \
  && apt-get clean && rm -rf /tmp/* /var/tmp/*

RUN pip3 install -U pip
RUN pip3 install --upgrade pip && \
    pip3 install torch==1.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
    pip3 install opencv-python==4.5.3.56 Flask==2.1.1 Flask-Cors==3.0.10 numpy==1.22.3 imageio==2.10.3

WORKDIR /app