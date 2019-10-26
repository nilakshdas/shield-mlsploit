FROM tensorflow/tensorflow

VOLUME /mnt/input
VOLUME /mnt/output

ADD requirements.txt /
RUN pip install -r /requirements.txt

COPY . /app
WORKDIR /app

CMD ["sh", "run.sh"]
