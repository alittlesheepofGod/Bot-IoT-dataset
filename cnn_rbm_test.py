# import libraries
from turtle import clear
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization, UpSampling2D
from sklearn.model_selection import train_test_split
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adagrad
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf 
from tensorflow.keras import regularizers
from keras.models import Model

# import libs for DA
import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  

from matplotlib import pyplot as plt 
import numpy as np 
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import BatchNormalization 
from keras.models import Sequential 
from tensorflow.keras.optimizers import Adadelta, RMSprop, SGD, Adam 
from keras import regularizers 
from tensorflow.keras.utils import to_categorical 


# import module handling
import handling 

X_train = handling.X_train
X_test = handling.X_test

y_train = handling.y_train
y_test = handling.y_test

# # change one-hot encoding to integer 
# y_train = np.argmax(y_train, axis=1)
# y_test = np.argmax(y_test, axis=1)

# resize 
X_train = np.resize(X_train, (X_train.shape[0], X_train.shape[1]))
X_test = np.resize(X_test, (X_test.shape[0], X_train.shape[1]))


# normalizer
scaler = Normalizer().fit(X_train)
X_train = scaler.transform(X_train)
scaler = Normalizer().fit(X_test)
X_test = scaler.transform(X_test)

# standard scaler 
s = StandardScaler()
s.fit(X_train)
X_train = s.transform(X_train)
X_test = s.transform(X_test)

# increase features for cnn to 72 features:
X_train = np.resize(X_train, (X_train.shape[0], 72))
X_test = np.resize(X_test, (X_test.shape[0], 72))

#newcode pca train 
pca=PCA(n_components=72)
pca.fit(X_train)
x_train_pca=pca.transform(X_train)
#newcode pca test
pca.fit(X_test)
x_test_pca=pca.transform(X_test) 

# increase features for cnn to 72 features:
X_train = np.resize(X_train, (X_train.shape[0], 72))
X_test = np.resize(X_test, (X_test.shape[0], 72))

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
import numpy as np
from keras.datasets import mnist

# rbm = BernoulliRBM(n_components = 72, n_iter = 20,
# learning_rate = 0.01,  verbose = True)
# logistic = LogisticRegression(C = 1.0)
# # train the classifier and show an evaluation report
# classifier = Pipeline([("rbm", rbm), ("logistic", logistic)])
# classifier.fit(X_train, y_train)
# print("RBM + LOGISTIC REGRESSION ON ORIGINAL DATASET")
# print(classification_report(y_test, classifier.predict(X_test)))
# # nudge the dataset and then re-evaluate
# print("RBM + LOGISTIC REGRESSION ON NUDGED DATASET")
# print(classification_report(y_test, classifier.predict(X_test)))


# define CNN model 

# import libraries
from turtle import clear
from keras.models import Sequential
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.layers import Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, Flatten, BatchNormalization, Dense
import plotly.express as px
import plotly.offline as pyo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
import keras
import os, re, time, math, tqdm, itertools
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adagrad
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# CNN model
def model():
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=(72, 1)))
    model.add(Conv1D(filters=128, kernel_size=1, activation='relu'))
    # adding a pooling layer
    model.add(MaxPooling1D(pool_size=(3), strides=1, padding='same'))
    model.add(Conv1D(filters=128, kernel_size=1, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=1, padding='same'))
    model.add(Conv1D(filters=128, kernel_size=1, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=1, padding='same'))
    model.add(Conv1D(filters=128, kernel_size=1, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=1, padding='same'))
    model.add(Conv1D(filters=128, kernel_size=1, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=1, padding='same'))
    model.add(Conv1D(filters=128, kernel_size=1, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=1, padding='same'))
    model.add(Conv1D(filters=128, kernel_size=1, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=1, padding='same'))
    model.add(Conv1D(filters=128, kernel_size=1, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=1, padding='same'))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    # model.add(Dense(4, activation='softmax'))  # number of node in dense layer represent for number of classes to classification
    model.add(Dense(3, activation='softmax'))  # number of node in dense layer represent for number of classes to classification

    opt = SGD(lr = 0.01, momentum = 0.9, decay = 0.01)
    # opt = Adagrad()

    model.compile(loss='binary_crossentropy', optimizer = opt, metrics=['accuracy'])
    return model




# model : RBM + CNN

rbm = BernoulliRBM(n_components = 72, n_iter = 20,
learning_rate = 0.01,  verbose = True)

cnn = model()

logistic = LogisticRegression(C = 1.0)
# train the classifier and show an evaluation report
classifier = Pipeline([("rbm", rbm), ("CNN", cnn)])
classifier.fit(X_train, y_train)



# change one-hot encoding to integer 
y_train = np.argmax(y_train, axis=1)
y_test = np.argmax(y_test, axis=1)

print("RBM + LOGISTIC REGRESSION ON ORIGINAL DATASET")
print(classification_report(y_test, classifier.predict(X_test)))
# nudge the dataset and then re-evaluate
print("RBM + LOGISTIC REGRESSION ON NUDGED DATASET")
print(classification_report(y_test, classifier.predict(X_test)))

