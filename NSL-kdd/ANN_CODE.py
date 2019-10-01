import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.optimizers import SGD
from mlxtend.preprocessing import one_hot
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler




def preprocess_dataset(ds_name):
    # Importing the dataset
    dataset = pd.read_csv(ds_name)
    
    #d=dataset.duplicated()
    #print(sum(d))
    
    #dataset.drop_duplicates(keep='first')
    
    X = dataset.iloc[:, :-2].values
    y = dataset.iloc[:, -2].values
    
    # 5 classes DoS,prob,r2l,u2r,normal
    for i in range(len(y)):
      if y[i] in ["back", "land", "neptune", "pod", "smurf", "teardrop","apache2","processtable","worm","udpstorm","mailbomb"]:
        y[i]="DoS"
        #print(y[i])
      elif y[i] in ["satan","ipsweep","nmap","portsweep","mscan","saint"]:
        y[i]="prob"
      elif y[i] in ["guess_passwd","guess_password","ftp_write","imap","phf","multihop","warezmaster","warezclient","spy","xlock","xsnoop","snmpguess","snmpgetattack","httptunnel","sendmail","named"]:
        y[i]="r2l"
      elif y[i] in ["buffer_overflow", "loadmodule", "rootkit", "perl","sqlattack", "xterm", "ps"]:
        y[i]="u2r"
        
    print(y[0:10])    
        
    X=np.copy(X[:,:])
    
    labelencoder_x=LabelEncoder()
    X[:, 1]=labelencoder_x.fit_transform(X[:, 1])
    X[:, 2]=labelencoder_x.fit_transform(X[:, 2])
    X[:, 3]=labelencoder_x.fit_transform(X[:, 3])
    
    
    Y=np.copy(y)
    labelencoder_y=LabelEncoder()
    
    #final Y for splitting
    Y=labelencoder_y.fit_transform(Y)
    
    onehotencoder= OneHotEncoder(categorical_features=[1,2,3])
    X=onehotencoder.fit_transform(X).toarray()
    
    
    Y=one_hot(Y) 
    
    # Feature Scaling
    
    sc = MinMaxScaler(feature_range = (0, 1))
    
    
    #final X for spliting
    X = sc.fit_transform(X)
    
    return X,Y


#from google.colab import drive
#drive.mount('/content/drive')


X_Train,Y_Train = preprocess_dataset('KDDTrain+.txt')

X_Test,Y_Test = preprocess_dataset('KDDTest2+.txt')






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


model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy']
             )

model.fit(X_Train,Y_Train,
          epochs=100,
          batch_size=200)



score = model.evaluate(X_Test, Y_Test, batch_size=200)

print(score[1])
