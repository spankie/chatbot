from flask import Flask, request
from predictor import predict, get_model


model, words, labels, data = get_model()

app = Flask(__name__)


@app.route('/predict')
def predict_iris():
    message = request.args.get("word")
    reply = predict(message, model, words, labels, data)
    print(message)
    return str(reply)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
