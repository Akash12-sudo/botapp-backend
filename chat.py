
import random
import torch
import json
from app.model import NeuralNetwork
from app.nltk_tut import tokenize, bag_of_words

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('app/intends.json', 'r') as f:
    intends = json.load(f)

FILE = "app/data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]


model = NeuralNetwork(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Buddy"


def get_response(msg):

    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob > 0.90:
        for intent in intends['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    responses = ["I don't understand", "What are you saying?"]
    return random.choices(responses) 
    



