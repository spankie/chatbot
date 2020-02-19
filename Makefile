build:
	docker build -t takem/chatbot .

run:
	docker run -it takem/chatbot

interactive:
	docker run -p 8888:8888 jupyter/tensorflow-notebook


build-server:
	docker build -t takem/chatbot-server . -f Dockerfile-dev

server:
	docker run -it -p 8000:8000 takem/chatbot-server
