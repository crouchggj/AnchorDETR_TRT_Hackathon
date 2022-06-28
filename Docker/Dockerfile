# -*- mode: dockerfile -*-
FROM nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04
MAINTAINER  bob(bob@test.net)

ENV LANG C.UTF-8
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# Install necessary packages
RUN sed -i 's@//archive.ubuntu@//mirrors.aliyun@' /etc/apt/sources.list
RUN rm -rf /etc/apt/sources.list.d/cuda.list && rm -rf /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-get update && apt-get install -y --no-install-recommends wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb

RUN apt update
RUN apt install tzdata openssh-server libssl-dev python3-pip cmake vim -y

ADD TensorRT-8.4.0.6 /usr/local/TensorRT-8.4.0.6
RUN ln -s /usr/local/TensorRT-8.4.0.6 /usr/local/TensorRT
RUN cd /usr/local/TensorRT/python && pip3 install tensorrt-8.4.0.6-cp38-none-linux_x86_64.whl

RUN pip3 install cuda-python nvidia-pyindex torch==1.8.0 torchvision==0.9.0 tqdm pycocotools scipy onnx-simplifier==0.3.5 packaging -q -i https://pypi.douban.com/simple
RUN pip3 install onnx-graphsurgeon -i https://pypi.douban.com/simple
RUN ldconfig

