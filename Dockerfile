FROM tensorflow/tensorflow:1.14.0-py3

COPY . .

RUN pip install numpy \
  tflearn \
  # tensorflow \
  nltk

RUN python -m nltk.downloader punkt

ENTRYPOINT [ "python", "./main.py" ]