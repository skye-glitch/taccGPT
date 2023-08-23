# FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
WORKDIR /etc/TACC_GPT

COPY ./requirements.txt ./
RUN apt-get update -y
RUN apt-get install -y python3-all
RUN python3 -m pip install -r requirements.txt
COPY . .
EXPOSE 9995

CMD [ "python3", "main.py", "--path", "./nlp_models/output_1.3b_epoch32_sqLength512/", "--max_new_tokens", "512", "--http_host", "0.0.0.0", "--http_port", "9995" ]