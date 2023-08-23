FROM python:3.9
WORKDIR /etc/backend

COPY requirements.txt ./
RUN apt-get update -y
RUN apt-get install -y python3-all
RUN python3 -m pip install -r requirements.txt
COPY . .
EXPOSE 9990

CMD ["python3","main.py"]