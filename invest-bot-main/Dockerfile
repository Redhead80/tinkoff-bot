FROM python:latest
MAINTAINER i@utkin80.ru
LABEL version='1.1'
RUN apt-get update && \
    apt-get install -y python3-pip && \
    pip install --upgrade pip && \
    python3 -m venv .venv
RUN mkdir /tinkoffbot/
WORKDIR /tinkoffbot/
COPY . /tinkoffbot/
RUN pip install -r requirements.txt
ENTRYPOINT ["python3", "./main.py"]