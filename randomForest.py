import numpy as np
from numpy import inf,nan
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from mlxtend.preprocessing import one_hot
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def random_forest_classifier(features, target):
    clf = RandomForestClassifier()
    clf.fit(features, target)
    return clf

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


X_train, y_train = preprocess_dataset(
            '/Users/m.salmanghazi/Downloads/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv')
X_train, X_test, y_train, y_test = train_test_split(X_train,y_train,test_size=0.3,shuffle=True)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#

from sklearn.metrics import accuracy_score
trained_model = random_forest_classifier(X_train, y_train)
predictions = trained_model.predict(X_test)


print("Train Accuracy :: ", accuracy_score(y_train, trained_model.predict(X_train)))
print("Test Accuracy  :: ", accuracy_score(y_test, predictions))

