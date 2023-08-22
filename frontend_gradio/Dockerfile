FROM pytorch/pytorch:latest
WORKDIR /etc/TACC_GPT

COPY ./requirements.txt ./
RUN pip install -r requirements.txt
COPY . .
EXPOSE 9991
RUN which python

CMD [ "python3", "main.py", "--path", "./nlp_models/output_1.3b_epoch32_sqLength512/", "--max_new_tokens", "512" ]