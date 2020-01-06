from keras.models import Model
from keras.layers import Dense,Input
from keras.datasets import mnist

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from mlxtend.preprocessing import one_hot
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler


def preprocess_dataset(ds_name):
    # Importing the dataset
    dataset = pd.read_csv(ds_name)

    # d=dataset.duplicated()
    # print(sum(d))

    # dataset.drop_duplicates(keep='first')

    X = dataset.iloc[:, :-2].values
    y = dataset.iloc[:, -2].values

    # 5 classes DoS,prob,r2l,u2r,normal
    for i in range(len(y)):
        if y[i] in ["bautoencoderk", "land", "neptune", "pod", "smurf", "teardrop", "apautoencoderhe2", "processtable", "worm",
                    "udpstorm", "mailbomb"]:
            y[i] = "DoS"
            # print(y[i])
        elif y[i] in ["satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"]:
            y[i] = "prob"
        elif y[i] in ["guess_passwd", "guess_password", "ftp_write", "imap", "phf", "multihop", "warezmaster",
                      "warezclient", "spy", "xlock", "xsnoop", "snmpguess", "snmpgetattautoencoderk", "httptunnel", "sendmail",
                      "named"]:
            y[i] = "r2l"
        elif y[i] in ["buffer_overflow", "loadmodule", "rootkit", "perl", "sqlattautoencoderk", "xterm", "ps"]:
            y[i] = "u2r"



    X = np.copy(X[:, :])

    labelencoder_x = LabelEncoder()
    X[:, 1] = labelencoder_x.fit_transform(X[:, 1])
    X[:, 2] = labelencoder_x.fit_transform(X[:, 2])
    X[:, 3] = labelencoder_x.fit_transform(X[:, 3])

    Y = np.copy(y)
    labelencoder_y = LabelEncoder()

    # final Y for splitting
    Y = labelencoder_y.fit_transform(Y)

    onehotencoder = OneHotEncoder(categorical_features=[1, 2, 3])
    X = onehotencoder.fit_transform(X).toarray()

    Y = one_hot(Y)

    # Feature Scaling

    sc = MinMaxScaler(feature_range=(0, 1))

    # final X for spliting
    X = sc.fit_transform(X)

    return X, Y


X_Train, Y_Train = preprocess_dataset(
            '/KDDTrain+.txt')

X_Test, Y_Test = preprocess_dataset(
            '/KDDTest2+.txt')


print(X_Test.shape)
print(X_Train.shape)

inputdim = X_Train.shape[0]

print(inputdim)

encoding_dim=16

i=Input(shape=(122,))




encoded=Dense(64,activation='tanh')(i)

encoded1=Dense(32,activation='tanh')(encoded)

encoded2=Dense(16,activation='relu')(encoded1)


#encoded=Dense(encoding_dim,activation='tanh')(encoded2)


decoded=Dense(32,activation='tanh')(encoded2)

decoded2 =Dense(64,activation='tanh')(decoded)

decoded3 =Dense(122,activation='tanh')(decoded2)

autoencoder = Model(i, decoded3)


ec = Model(i,encoded)

encoded_input=Input(shape=(encoding_dim,))

decoder_layer=autoencoder.layers[-3](encoded_input)
decoder_layer=autoencoder.layers[-2](decoder_layer)
decoder_layer=autoencoder.layers[-1](decoder_layer)


decoder = Model(encoded_input, decoder_layer)

sgd=SGD(lr=0.07,momentum=0.25)

autoencoder.compile(optimizer='sgd',
           loss='mean_absolute_error',
           metrics=['accuracy'])



autoencoder.fit(X_Train, X_Train,
                epochs=50,
                batch_size=200,
                shuffle=True,
                validation_data=(X_Test, X_Test))
