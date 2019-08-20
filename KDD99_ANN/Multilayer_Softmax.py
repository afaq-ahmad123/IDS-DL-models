import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.optimizers import SGD
from mlxtend.preprocessing import one_hot
import pandas as pd
import numpy as np

dataset = pd.read_csv('kddcup.data_10_percent_corrected')
X = dataset.iloc[:, :-1].values     #selects all col except last
y = dataset.iloc[:, -1].values      #selects last col only



# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
Port_encoder = LabelEncoder()
X[:, 1 ] = Port_encoder.fit_transform(X[:, 1])

Protocol_encoder = LabelEncoder()
X[:, 2 ] = Protocol_encoder.fit_transform(X[: , 2])

Flag_encoder = LabelEncoder()
X[:, 3] =Flag_encoder.fit_transform(X[: , 3])

onehotencoder = OneHotEncoder(categorical_features=[1,2,3])
X = onehotencoder.fit_transform(X).toarray()


output_encoder = LabelEncoder()
y =output_encoder.fit_transform(y)

#onehotencoder2 = OneHotEncoder()
y=one_hot(y)        #mlxtend lib one_hots the 1d array




# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

input_dim=X_train.shape[1]
no_of_classes = y_train.shape[1]


#adding layers
model = Sequential()
model.add(Dense(64,activation='relu',input_dim=input_dim))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(no_of_classes,activation='softmax'))

#stochastic_gradient_descent
sgd=SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)

model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy']
             )

model.fit(X_train,y_train,
          epochs=1000,
          batch_size=128)
