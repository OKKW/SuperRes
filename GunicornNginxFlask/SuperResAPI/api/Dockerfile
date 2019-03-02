# FROM python:3.6
FROM ubuntu:latest
LABEL MAINTANER "oskar.wilhelmsson@acando.com"

RUN apt-get update -y
RUN apt-get install -y python-pip python3-pip python-dev build-essential

COPY . /app
WORKDIR /app

RUN pip3 install redis
RUN pip3 install gunicorn
RUN pip3 install -r requirements.txt
RUN pip3 install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
RUN pip3 install torchvision



#ENTRYPOINT ["python3"]
#EXPOSE 5000
#CMD ["app.py"]