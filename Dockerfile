FROM nvcr.io/nvidia/pytorch:23.07-py3

## config
ARG USER=wangsadirdja
ARG UID=1174

RUN mkdir -p /home/stud/${USER}
RUN adduser stud${USER} --uid ${UID} --home /home/stud/${USER}/ --disabled-password --gecos "" --no-create-home

WORKDIR /home/stud/${USER}
COPY requirements.txt /home/stud/${USER}
RUN apt-get update && apt-get install -y git
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

RUN chown -R stud${USER} /home/stud/${USER}

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# NAME=ls6-stud-registry.informatik.uni-wuerzburg.de/studwangsadirdja-nlpss23:0.0.1
