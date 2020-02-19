import tensorflow
import tflearn
import random
import pickle
import numpy
import nltk
import json
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def get_model():
    with open("intents.json") as file:
        data = json.load(file)
    with open("data/data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
    tensorflow.reset_default_graph()

    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)

    model = tflearn.DNN(net)
    model.load("data/model.tflearn")
    return (model, words, labels, data)


def predict(inp, model, words, labels, data):
    # predict the word.
    # inp is the word to predict a reply to
    # words
    results = model.predict([bag_of_words(inp, words)])[0]
    # at the point,the output is a probability
    results_index = numpy.argmax(results)
    # at this point, the output is index of the greatest possible result from the probability
    tag = labels[results_index]
    # at this point, the output is the tag which the index belongs to
    if results[results_index] > 0.6:
        # this if statement determines how high the percentage of the probability must be to accepted
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        reply = random.choices(responses)[0]
    else:
        reply = "I do not understand that. Could you ask another question?"

    return reply
