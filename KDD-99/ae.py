
import pandas as pd
import torch as tc
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

import numpy as np


# Importing the dataset
training_set = pd.read_csv('/Users/m.salmanghazi/Downloads/kddcup.data.corrected.csv')

test_set = pd.read_csv('/Users/m.salmanghazi/Downloads/kddcup.data_10_percent_corrected.csv')
training_set = training_set.iloc[:, :].values
test_set = test_set.iloc[:, :].values


#
from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
training_set[:, 1]=labelencoder_x.fit_transform(training_set[:, 1])  # protocol type
test_set[:, 1]=labelencoder_x.fit_transform(test_set[:, 1])
training_set[:, 2]=labelencoder_x.fit_transform(training_set[:, 2])  # service
test_set[:, 2]=labelencoder_x.fit_transform(test_set[:, 2])
training_set[:, 3]=labelencoder_x.fit_transform(training_set[:, 3])  # service
test_set[:, 3]=labelencoder_x.fit_transform(test_set[:, 3])
training_set[:, -1]=labelencoder_x.fit_transform(training_set[:, -1])  # result
test_set[:, -1]=labelencoder_x.fit_transform(test_set[:, -1])

test_set = np.array(test_set, dtype=int)
training_set = np.array(training_set, dtype=int)


protocole_type = int(max(max(training_set[:,1]), max(test_set[:,1])))
service = int(max(max(training_set[:,2]), max(test_set[:,2])))


def convert(data):
    new_data = []
    for pt in range(1, protocole_type + 1):
        id_s = data[:, 2][data[:, 1] == pt]

        id_r = data[:, -1][data[:, 1] == pt]
        ratings = np.zeros(service)
        ratings[id_s - 1] = id_r
        new_data.append(list(ratings))
        return new_data


training_set = convert(training_set)
test_set = convert(test_set)

# # Converting the data into Torch tensors
training_set = tc.FloatTensor(training_set)
test_set = tc.FloatTensor(test_set)


# creating NN
class SAE(nn.Module):
    def __init__ (self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(service, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, service)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x


sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5)

# TRAINING...
nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for i in range(protocole_type):
        input = Variable(training_set).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            # loss = CrossEntropyLoss()
            # loss(Pip, Train["Label"])
            loss = criterion(output, target)
            mean_corrector = service/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.item()*mean_corrector)
            s += 1.
            optimizer.step()
    print("Epoch no. " + str(epoch) + " LOSS " + str(train_loss/s))

#TESTING
test_loss = 0
s = 0.
for i in range(protocole_type):
    input = Variable(training_set).unsqueeze(0)
    target = input.clone()
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        # loss = CrossEntropyLoss()
        # loss(Pip, Train["Label"])
        loss = criterion(output, target)
        mean_corrector = service / float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.item() * mean_corrector)
        s += 1.
        optimizer.step()
print()
print()
print("TEST LOSS " + str(train_loss/s))