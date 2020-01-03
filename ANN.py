
from numpy import inf,nan

from keras.models import Model
from keras.layers import Dense,Input

from keras.models import Sequential

import numpy as np

import pandas as pd

from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD


from mlxtend.preprocessing import one_hot
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler


def preprocess_dataset(ds_name):
    # Importing the dataset
    dataset = pd.read_csv(ds_name)

    dataset = dataset[dataset.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]


    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # attack types: ['BENIGN' 'DoS GoldenEye' 'DoS Hulk' 'DoS Slowhttptest' 'DoS slowloris' 'Heartbleed']


    #X = np.copy(X[:, :])


    Y = np.copy(y)
    labelencoder_y = LabelEncoder()

    # final Y for splitting
    Y = labelencoder_y.fit_transform(Y)
    Y = one_hot(Y)

    # Feature Scaling
    sc = MinMaxScaler(feature_range=(0, 1))

    # final X for spliting

    X[X == -inf] = 0
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if float(X[i,j]) >= 1.7976931348623157e+308:
                X[i,j] = 1.7976931348623157e+308

    X[X == nan] = 1

    X = sc.fit_transform(X)

    return X, Y


X_Train, Y_Train = preprocess_dataset(
            '/Users/m.salmanghazi/Downloads/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv')
X_Test, Y_Test = preprocess_dataset(
            '/Users/m.salmanghazi/Downloads/MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv')






input_dim=X_Train.shape[1]
no_of_classes = Y_Train.shape[1]


#adding layers
model = Sequential()

model.add(Dense(64,activation='sigmoid',input_dim=input_dim))
model.add(Dropout(0.8))
model.add(Dense(64,activation='sigmoid'))
model.add(Dropout(0.8))
model.add(Dense(64,activation='sigmoid'))
#model.add(Dropout(0.8))


#output layer
model.add(Dense(no_of_classes,activation='softmax'))

#stochastic_gradient_descent
sgd=SGD(lr=0.05,decay=1e-6,momentum=0.9,nesterov=True)


model.compile(loss='mean_squared_logarithmic_error',
              optimizer='rmsprop',
              metrics=['accuracy']
             )

model.fit(X_Train,Y_Train,
          epochs=100,
          batch_size=200)



score = model.evaluate(X_Test, Y_Test, batch_size=200)

print(score[1])