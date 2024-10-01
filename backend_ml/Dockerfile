# FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
# FROM pytorch/pytorch:latest

# FROM python:3.9
WORKDIR /etc/TACC_GPT


COPY ./requirements.txt ./
RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt