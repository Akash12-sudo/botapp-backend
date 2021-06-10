
import json
from app.nltk_tut import tokenize, stem, bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from app.model import NeuralNetwork

with open('app/intends.json', 'r') as f:

    intends = json.load(f)

all_words = []
tags = []
xy = []

for intent in intends['intents']:

    tag = intent['tag']
    tags.append(tag)

    for pattern in intent['patterns']:

        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '.', ',', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# print(all_words)

all_words = sorted(set(all_words))
tags = sorted(set(tags))

# print(all_words)
# print(tags)

X_train = []
y_train = []

for (pattern_sentence, tag) in xy:

    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)


class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Hyperparameters
batch_size = 8
input_size = len(X_train[0])
hidden_size = 15
output_size = len(tags)
learning_rate = 0.002
epochs = 1000

# print(input_size, len(all_words))
# print(output_size, tags)

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNetwork(input_size, hidden_size, output_size).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        # forward pass
        output = model(words)
        loss = criterion(output, labels)

        # backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch +1)%100 == 0:
        print(f'epoch {epoch+1}/{epochs}, loss={loss.item():.4f}')


print(f'final loss, loss={loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "app/data.pth"
torch.save(data, FILE)

print(f'training complete. File saved at {FILE}')
