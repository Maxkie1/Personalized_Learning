# syntax=docker/dockerfile:1
FROM python:3.10-slim-buster

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/* 

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "-u", "main.py", "--poll", "4"]