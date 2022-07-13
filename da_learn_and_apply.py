import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"   # model will be trained on GPU 0 
import keras 
from matplotlib import pyplot as plt 
import numpy as np 
import gzip 
from keras.models import Model 
from tensorflow.keras.optimizers import RMSprop 
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras.layers import BatchNormalization 
from keras.models import Model
from tensorflow.keras.optimizers import RMSprop 
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA

import handling 

# data 
train_data = handling.X_train

# Shapes of training set 
print("Training set (images) shape: {shape}".format(shape=train_data.shape))

# train label 
train_labels = handling.y_train

# data preprocessing 

# resize 
X_train = np.resize(train_data, (train_data.shape[0], train_data.shape[1]))

# normalizer
scaler = Normalizer().fit(X_train)
X_train = scaler.transform(X_train)

# standard scaler 
s = StandardScaler()
s.fit(X_train)
X_train = s.transform(X_train)

# increase features for cnn to 72 features:
X_train = np.resize(X_train, (X_train.shape[0], 72))

#newcode pca train 
pca=PCA(n_components=72)
pca.fit(X_train)
x_train_pca=pca.transform(X_train)

# increase features for cnn to 72 features:
X_train = np.resize(X_train, (X_train.shape[0], 72, 1))

X_train = X_train / np.max(X_train)
# model 

# partition the data : 80% training and 20% for validation 

from sklearn.model_selection import train_test_split
train_X, valid_X, train_ground, valid_ground = train_test_split(train_data, train_data, test_size=0.2)

batch_size = 64
epochs = 10 
inChannel = 1
x =  72
input = Input(shape = (x, inChannel))
num_classes = 10

def encoder(input):
    # encoder 
    # encoder 
    conv1 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same', input_shape=(72, 1))(input) # 28x28x32
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv1D(filters=32, kernel_size=1, activation='relu')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling1D(pool_size=(3), strides=1, padding='same')(conv1) #14 x 14 x 32
    conv2 = Conv1D(filters=64, kernel_size=1, activation='relu')(conv1) #14 x 14 x 64
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv1D(filters=64, kernel_size=1, activation='relu')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling1D(pool_size=(3), strides=1, padding='same')(conv2) #7 x 7 x 64
    conv3 = Conv1D(filters=128, kernel_size=1, activation='relu')(conv2) #7 x 7 x 128 (small and thick)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv1D(filters=128, kernel_size=1, activation='relu')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv1D(filters=256, kernel_size=1, activation='relu')(conv3) #7 x 7 x 256 (small and thick)
    conv4 = BatchNormalization()(conv4)
    # conv4 = Conv1D(filters=256, kernel_size=1, activation='relu')(conv4)
    # conv4 = BatchNormalization()(conv4)
    return conv4

def decoder(conv4):
    # decoder
    conv5 = Conv1D(filters=128, kernel_size=1, activation='relu')(conv4) #7 x 7 x 128
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv1D(filters=128, kernel_size=1, activation='relu')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv6 = Conv1D(filters=64, kernel_size=1, activation='relu')(conv5) #7 x 7 x 64
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv1D(filters=64, kernel_size=1, activation='relu')(conv6)
    conv6 = BatchNormalization()(conv6)
    up1 = UpSampling1D(size=(3))(conv6) #14 x 14 x 64
    conv7 = Conv1D(filters=32, kernel_size=1, activation='relu')(conv6) # 14 x 14 x 32
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv1D(filters=32, kernel_size=1, activation='relu')(conv7)
    conv7 = BatchNormalization()(conv7)
    # up2 = UpSampling1D(size=(3))(conv7) # 28 x 28 x 32
    decoded = Conv1D(filters=1, kernel_size=1, activation='sigmoid')(conv7) # 28 x 28 x 1
    return decoded

autoencoder = Model(input, decoder(encoder(input)))
autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())

autoencoder.summary()

# train model 
autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_ground))

# plot loss function : training loss and validation loss 
loss = autoencoder_train.history['loss']
val_loss = autoencoder_train.history['val_loss']
epochs = range(10)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
