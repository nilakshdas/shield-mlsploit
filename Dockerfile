FROM tensorflow/tensorflow:2.2.0rc1-py3

VOLUME /mnt/input
VOLUME /mnt/output

RUN apt-get update && apt-get install -y git

COPY requirements.txt /
RUN pip install -r /requirements.txt

COPY . /app
WORKDIR /app

CMD ["sh", "run.sh"]
