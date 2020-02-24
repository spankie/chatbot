#!/bin/sh

pip3 install flask tensorflow==1.14.0 numpy tflearn nltk

pip3 install -U flask-cors

python3 -m nltk.downloader punkt