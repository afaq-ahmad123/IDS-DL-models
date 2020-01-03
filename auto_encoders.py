
from numpy import inf,nan

from keras.models import Model
from keras.layers import Dense,Input


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
print(X_Train.shape)
print(Y_Train.shape)


inputdim = X_Train.shape[0]

print(inputdim)

encoding_dim=10

i=Input(shape=(78,))




encoded=Dense(40,activation='sigmoid')(i)

encoded1=Dense(20,activation='sigmoid')(encoded)

encoded2=Dense(10,activation='relu')(encoded1)


#encoded=Dense(encoding_dim,activation='sigmoid')(encoded2)


decoded=Dense(20,activation='sigmoid')(encoded2)

decoded2 =Dense(40,activation='sigmoid')(decoded)

decoded3 =Dense(78,activation='sigmoid')(decoded2)

autoencoder = Model(i, decoded3)


ec = Model(i,encoded)

encoded_input=Input(shape=(encoding_dim,))

decoder_layer=autoencoder.layers[-3](encoded_input)
decoder_layer=autoencoder.layers[-2](decoder_layer)
decoder_layer=autoencoder.layers[-1](decoder_layer)


decoder = Model(encoded_input, decoder_layer)

sgd=SGD(lr=0.07,momentum=0.35)

autoencoder.compile(optimizer='rmsprop',
           loss='mean_squared_logarithmic_error',
           metrics=['accuracy'])



autoencoder.fit(X_Train, X_Train,
                epochs=50,
                batch_size=200,
                shuffle=True,
                validation_data=(X_Test, X_Test))
