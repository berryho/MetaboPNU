from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn import preprocessing


import pandas as pd
import numpy as np
import os
import tensorflow as tf



# seed
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# data
df = pd.read_csv('../dataset/Ginseng-total.csv')

dataset = df.values
X_ = dataset[:,2:568]
Y_obj = dataset[:,1]
Z_id = dataset[:,0]
X = preprocessing.normalize(X_)

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)
Y_encoded = np_utils.to_categorical(Y)

# train set, test set
X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(X, Y_encoded, Z_id,
                                                                     test_size=0.3, random_state=seed)

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# model
model = Sequential()
model.add(Dense(450, input_dim=X.shape[1]))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Dense(450))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.1))


model.add(Dense(3))
model.add(BatchNormalization())
model.add(Activation('softmax'))
model.summary()



# model compile
model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

# save model
MODEL_DIR = './ginseng_model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath="./ginseng_model/{epoch:02d}-{val_loss:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=0, save_best_only=True)

# Train
hist = model.fit(X_train, Y_train, epochs=930, batch_size=95, validation_data=(X_test, Y_test),
                 verbose=1, callbacks=[checkpointer])
