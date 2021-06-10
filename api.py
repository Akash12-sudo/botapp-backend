
import time
from flask import Flask, request, jsonify
from app.chat import bot_name, get_response
import json 
from flask.json import jsonify, dumps
from flask_cors import CORS
import pyttsx3
from pyttsx3 import *
import speech_recognition as sr
import random
import os

app = Flask(__name__, static_folder='../frontend/build')
CORS(app)

engine = pyttsx3.init('dummy')

@app.route('/greet')
def greet():
    text = "Welcome to your own assistant Chatbot. How can I help you?"
    engine.setProperty('voice', 'english+f2')
    engine.setProperty("rate", 170)
    engine.say(text)
    engine.runAndWait()
    return {'response': "greet"}



@app.route('/response', methods=["POST"])
def get_resp():
    resp = dumps(request.json)
    if resp == "":
        return {'response': "I don't understand"}

    text = get_response(resp)
    return {'response': text};


@app.route('/speech', methods=["POST"])
def get_speech():

    req = dumps(request.json)
    # engine.setProperty('voice', 'english+f2')
    engine.setProperty("rate", 170)
    engine.say(req)
    engine.runAndWait()
    return {'response': "Hello"};


@app.route('/', methods=['GET'])
def hello():
    return jsonify({"resp":"This is a AI Chatbot Project"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", threaded=True, port=5000)

