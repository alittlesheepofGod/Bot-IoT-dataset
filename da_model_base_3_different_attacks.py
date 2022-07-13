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
X_train = np.resize(X_train, (X_train.shape[0], 72, 1))
X_test = np.resize(X_test, (X_test.shape[0], 72, 1))

# parition the data : 80% training and 20% for validation

from sklearn.model_selection import train_test_split
train_X, valid_X, train_ground, valid_ground = train_test_split(X_train, X_train, test_size=0.2)

# DA model
def encoder(input):
    # encoder 
    # encoder 
    conv1 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_uniform', input_shape=(72, 1))(input) # 28x28x32
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer='he_uniform', activity_regularizer=regularizers.l1(1e-4))(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling1D(pool_size=(3), strides=1, padding='same')(conv1) #14 x 14 x 32
    conv2 = Conv1D(filters=64, kernel_size=1, activation='relu', kernel_initializer='he_uniform', activity_regularizer=regularizers.l1(1e-4))(pool1) #14 x 14 x 64
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv1D(filters=64, kernel_size=1, activation='relu', kernel_initializer='he_uniform', activity_regularizer=regularizers.l1(1e-4))(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling1D(pool_size=(3), strides=1, padding='same')(conv2) #7 x 7 x 64
    conv3 = Conv1D(filters=128, kernel_size=1, activation='relu', kernel_initializer='he_uniform', activity_regularizer=regularizers.l1(1e-4))(pool2) #7 x 7 x 128 (small and thick)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv1D(filters=128, kernel_size=1, activation='relu', kernel_initializer='he_uniform', activity_regularizer=regularizers.l1(1e-4))(conv3)
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv1D(filters=256, kernel_size=1, activation='relu', kernel_initializer='he_uniform', activity_regularizer=regularizers.l1(1e-4))(conv3) #7 x 7 x 256 (small and thick)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv1D(filters=256, kernel_size=1, activation='relu', kernel_initializer='he_uniform', activity_regularizer=regularizers.l1(1e-4))(conv4)
    conv4 = BatchNormalization()(conv4)
    return conv4

def decoder(conv4):
    # decoder
    conv5 = Conv1D(filters=128, kernel_size=1, activation='relu', kernel_initializer='he_uniform', activity_regularizer=regularizers.l1(1e-4))(conv4) #7 x 7 x 128
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv1D(filters=128, kernel_size=1, activation='relu', kernel_initializer='he_uniform', activity_regularizer=regularizers.l1(1e-4))(conv5)
    conv5 = BatchNormalization()(conv5)
    conv6 = Conv1D(filters=64, kernel_size=1, activation='relu', kernel_initializer='he_uniform', activity_regularizer=regularizers.l1(1e-4))(conv5) #7 x 7 x 64
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv1D(filters=64, kernel_size=1, activation='relu', kernel_initializer='he_uniform', activity_regularizer=regularizers.l1(1e-4))(conv6)
    conv6 = BatchNormalization()(conv6)
    up1 = UpSampling2D((2,2))(conv6) #14 x 14 x 64
    conv7 = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer='he_uniform', activity_regularizer=regularizers.l1(1e-4))(up1) # 14 x 14 x 32
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv1D(filters=32, kernel_size=1, activation='relu', kernel_initializer='he_uniform', activity_regularizer=regularizers.l1(1e-4))(conv7)
    conv7 = BatchNormalization()(conv7)
    up2 = UpSampling2D((2,2))(conv7) # 28 x 28 x 32
    decoded = Conv1D(filters=1, kernel_size=1, activation='sigmoid', kernel_initializer='he_uniform', activity_regularizer=regularizers.l1(1e-4))(up2) # 28 x 28 x 1
    return decoded

autoencoder = Model(train_X, decoder(encoder(train_X)))

autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())
autoencoder.summary()

# train model 
# autoencoder_train = autoencoder.fit(X_train, y_train, batch_size=64,epochs=10,verbose=1,validation_data=(X_test, y_test))






# # plot loss function : training loss and validation loss 
# loss = autoencoder_train.history['loss']
# val_loss = autoencoder_train.history['val_loss']
# epochs = range(10)
# plt.figure()
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()

# # save model 
# autoencoder.save_weights('autoencoder.h5')

# # Segmenting the Fashion Mnist Images 

# # Change the labels from categorical to one-hot encoding 
# train_Y_one_hot = to_categorical(train_labels)
# test_Y_one_hot = to_categorical(test_labels)

# # Display the change for category label using one-hot encoding 
# print('Original label:', train_labels[0])
# print('After conversion to one-hot:', train_Y_one_hot[0])


# # Visualization of Results (CNN)
# # Let's make a graphical visualization of results obtained by applying CNN to our data 
# scores = model.evaluate(X_test, y_test, verbose = 1)
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

# # epochs = range(1, len(history['loss']) + 1)
# # acc = history['accuracy']
# # loss = history['loss']
# # val_acc = history['val_accuracy']
# # val_loss = history['val_loss']

# # draw configure matrix 
# from sklearn.metrics import ConfusionMatrixDisplay
# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# import numpy as np
# y_pred = model.predict(X_test)
# # labels = ["Benign", "DoS attacks-GoldenEye", "DoS attacks-Slowloris"]
# # labels = ["Benign", "DoS attacks-SlowHTTPTest", "DoS attacks-Hulk"]
# # labels = ["Benign", "DDOS attacks-LOIC-UDP", "DDOS attack-HOIC"]
# # labels = ["Benign", "Brute Force -Web", "Brute Force -XSS", "SQL Injection"]
# # labels = ["Benign", "Bot"]
# # labels = ["Benign", "Infilteration"]
# labels = ["Normal", "DDOS_HTTP"]

# # convert to categorical 
# from keras.utils.np_utils import to_categorical
# y_predict = to_categorical(np.argmax(y_pred, 1), dtype="int64")
# # convert one-hot encoding to integer
# y_predict = np.argmax(y_predict, axis=1)
# y_test = np.argmax(y_test, axis=1)
# cm = confusion_matrix(y_test, y_predict)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
# disp.plot(cmap=plt.cm.Blues)
# plt.show()

# # visualize training and val accuracy
# # plt.figure(figsize=(10, 5))
# # plt.title('Training and Validation Loss (CNN)')
# # plt.xlabel('Epochs')
# # plt.ylabel('Loss')
# # plt.plot(epochs, loss, label='loss', color='g')
# # plt.plot(epochs, val_loss, label='val_loss', color='r')
# # plt.legend()

# # list all data in history
# print(history.history.keys())
# # summarize history for accuracy
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# # print out accuracy for each class
# matrix = confusion_matrix(y_test, y_predict)
# # print("accuracy of benign, DoS attacks-SlowHTTPTest, DoS attacks-Hulk")
# # print("accuracy of benign, DoS attacks-LOIC-HTTP")
# # print(" accuracy of Benign, Brute Force -Web, Brute Force -XSS, SQL Injection: ") 
# print(" accuracy of Normal, DDOS_HTTP: ") 
# print(matrix.diagonal()/matrix.sum(axis=1))

# # print out False Alarm Rate 
# print("False Alarm Rate is : ")
# FAR = 0
# for i in range(1, len(cm[0])):
#     FAR += cm[0][i]
# FAR = FAR / (cm[0][0] + FAR)
# print(FAR)

# # print detection rate
# print("Detection Rate is : ")
# DTrate = 0
# for i in range(1, len(cm)):
#     for j in range(0, len(cm[i])):
#         DTrate += cm[i][j]

# sum = 0
# for i in range(1, len(cm)):
#     sum += cm[i][i]

# DTrate = sum / DTrate 

# print(DTrate)

# # Conclusion after CNN Training 

# """
# After training our deep CNN model on training data and validating it on validation data, it can be 
# interpreted that:

# + Model was trained on 10 epochs
# + CNN performed exceptionally well on training data and the accuracy was 99%
# + Model accuracy was down to 83.55% on validation data after 50 iterations, and gave a good accuracy
# of 92% after 30 iterations. Thus, it can be interpreted that optimal number of iterations on which this
# model can perform are 30. --> ... 


# """

