# FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
# FROM pytorch/pytorch:latest

# FROM python:3.9
WORKDIR /etc/TACC_GPT

COPY ./requirements.txt ./
RUN python3 -m pip install --upgrade pip
# RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt
COPY . .
EXPOSE 9990

# output_opt-2.7b_epoch16_train_batch_size8_seq512_lora256_zero1
CMD [ "python3", "main.py", "--path", "./nlp_models/output_1.3b_epoch32_sqLength512/", "--max_new_tokens", "512", "--http_host", "0.0.0.0", "--http_port", "9990" ]
# CMD [ "python3", "main.py", "--path", "./nlp_models/output_opt-2.7b_epoch16_train_batch_size8_seq512_lora256_zero1/", "--max_new_tokens", "512", "--http_host", "0.0.0.0", "--http_port", "9990" ]