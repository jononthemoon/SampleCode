# -*- coding: utf-8 -*-
"""ML-project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11mVe7Oy6g_1reNPTpVeeek8sph5MQuOO


This project was a classification task on sequential data about solar flares. The main task was predicting one of 5 classes for the solar flares, the 4th and 5th being extremely imbalanced.
I attempted a couple of things to deal with the imbalanced dataset, including collapsing the classes to have 0-3 be 0 and 4-5 be 1.

Mounting drive.
"""

from google.colab import drive
drive.mount('/content/drive')

"""Importing libraries."""

from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, GlobalMaxPooling1D
import keras
import numpy as np
import math 
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import tensorflow as tf
import os

tf.keras.backend.clear_session()

resolver = tf.distribute.cluster_resolver.TPUClusterResolver('grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)

# This is the TPU initialization code that has to be at the beginning.
tf.tpu.experimental.initialize_tpu_system(resolver)
print("All devices: ", tf.config.list_logical_devices('TPU'))

strategy = tf.distribute.TPUStrategy(resolver)

"""Importing data set as 3 folds and concatenating y.

"""

data_path = '/content/drive/My Drive/Datasets/solar-flares/'  # Use your own path here
k1 = np.load(data_path+'gsu_flares_znorm_part_1.npy').reshape(-1, 33*60)
k2 = np.load(data_path+'gsu_flares_znorm_part_2.npy').reshape(-1, 33*60)
k3 = np.load(data_path+'gsu_flares_znorm_part_3.npy').reshape(-1, 33*60)
y1 = np.load(data_path+'labels_1.npy')
y2 = np.load(data_path+'labels_2.npy')
y3 = np.load(data_path+'labels_3.npy')
y = np.concatenate((y1, y2, y3))

"""#Part 1: Keeping 5 Classes

##Employing RandomForestClassifier.
"""

model = RandomForestClassifier(n_jobs = -1)
y_pred = np.zeros(y1.shape[0] + y2.shape[0] + y3.shape[0])

model.fit(np.vstack((k2, k3)), np.append(y2, y3))
y_pred[:k1.shape[0]] = model.predict(k1)

model.fit(np.vstack((k1, k3)), np.append(y1, y3))
y_pred[k1.shape[0]:k1.shape[0] + k2.shape[0]] = model.predict(k2)

model.fit(np.vstack((k1, k2)), np.append(y1, y2))
y_pred[k1.shape[0] + k2.shape[0]:] = model.predict(k3)

print(classification_report(y, y_pred))

"""##Employing Conv1D

Prep data for Conv1D
"""

k1 = k1.reshape(-1, 33, 60)
k2 = k2.reshape(-1, 33, 60)
k3 = k3.reshape(-1, 33, 60)

"""Create Conv1D Model."""

model = Sequential()
model.add(Conv1D(128, 11, activation="relu", input_shape=(33, 60)))
model.add(MaxPooling1D(2, padding = 'same'))
model.add(Conv1D(64, 11, activation = 'relu'))
model.add(MaxPooling1D(2, padding = 'same'))
model.add(Flatten())
model.add(Dense(1000, activation = 'relu'))
model.add(Dense(500, activation = 'relu'))
model.add(Dense(5, activation = 'softmax'))
optimizer = keras.optimizers.Adam(lr=0.001)
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
model.summary()

"""K = 3 Folds Prep"""

y_pred = np.zeros((y.shape[0]), dtype="int32")

####1st Fold
x_1st_fold = np.vstack((k2, k3))
print(x_1st_fold.shape)
y_1st_fold = np.append(y2, y3)
print(y_1st_fold.shape)
y_1st_fold = tf.keras.utils.to_categorical(y_1st_fold, 5)
print(y_1st_fold.shape)

####2nd Fold
x_2nd_fold = np.vstack((k1, k3))
print(x_2nd_fold.shape)
y_2nd_fold = np.append(y1, y3)
print(y_2nd_fold.shape)
y_2nd_fold = tf.keras.utils.to_categorical(y_2nd_fold, 5)
print(y_2nd_fold.shape)

###3rd Fold
x_3rd_fold = np.vstack((k1, k2))
print(x_3rd_fold.shape)
y_3rd_fold = np.append(y1, y2)
y_3rd_fold = tf.keras.utils.to_categorical(y_3rd_fold, 5)
print(y_3rd_fold.shape)

"""Model Fitting and Predicting"""

weights = {0:1, 1:2, 2:5, 3:50, 4:100}
es = keras.callbacks.EarlyStopping(monitor = 'loss', patience = 10)
model.fit(x_1st_fold, y_1st_fold, batch_size = 32, epochs = 30, callbacks = [es], class_weight= weights)
y_pred[:k1.shape[0]] = np.argmax(model.predict(k1), axis=-1)
model.fit(x_2nd_fold, y_2nd_fold, batch_size = 32, epochs = 30, callbacks = [es], class_weight= weights)
y_pred[k1.shape[0]:k1.shape[0] + k2.shape[0]] = np.argmax(model.predict(k2), axis=-1)
model.fit(x_3rd_fold, y_3rd_fold, batch_size = 32, epochs = 30, callbacks = [es], class_weight= weights)
y_pred[k1.shape[0] + k2.shape[0]:] = np.argmax(model.predict(k3), axis=-1)

"""Evaluate results."""

print(classification_report(y, y_pred))

model = tf.keras.models.Sequential()
model.add(keras.layers.GRU(32, 
                     dropout = 0.1, 
                     recurrent_dropout = 0.5, 
                     return_sequences = True, 
                     input_shape = (33, 60)))
model.add(keras.layers.GRU(64, activation = 'relu',
                     dropout = 0.1,
                     recurrent_dropout = 0.5,
                     return_sequences = True))
model.add(keras.layers.GRU(128, activation = 'relu',
                     dropout = 0.1,
                     recurrent_dropout = 0.5,
                     return_sequences = True))
model.add(keras.layers.GRU(256, activation = 'relu',
                     dropout = 0.1,
                     recurrent_dropout = 0.5))
model.add(keras.layers.Dense(5, activation = 'softmax'))
optimizer = keras.optimizers.Adam(lr=0.0001)
metrics = [
          keras.metrics.TruePositives(name='tp'),
          keras.metrics.FalsePositives(name='fp'),
          keras.metrics.TrueNegatives(name='tn'),
          keras.metrics.FalseNegatives(name='fn'),
]
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = metrics)

weights = {0:1, 1:1, 2:1, 3:1, 4:1}
es = keras.callbacks.EarlyStopping(monitor = 'loss', patience = 3)
model.fit(x_1st_fold, y_1st_fold, batch_size = 32, epochs = 10, callbacks = [es], class_weight= weights)
y_pred[:k1.shape[0]] = np.argmax(model.predict(k1), axis=-1)
model.fit(x_2nd_fold, y_2nd_fold, batch_size = 32, epochs = 10, callbacks = [es], class_weight= weights)
y_pred[k1.shape[0]:k1.shape[0] + k2.shape[0]] = np.argmax(model.predict(k2), axis=-1)
model.fit(x_3rd_fold, y_3rd_fold, batch_size = 32, epochs = 10, callbacks = [es], class_weight= weights)
y_pred[k1.shape[0] + k2.shape[0]:] = np.argmax(model.predict(k3), axis=-1)

print(classification_report(y, y_pred))

"""#Part 2: Collapsing Classes into 0-2 = 0, 4-5 = 1

Collapsing classes 0-2 into class 0 and 3-4 into class 1.
"""

y1c = np.load(data_path+'labels_1.npy')
y2c = np.load(data_path+'labels_2.npy')
y3c = np.load(data_path+'labels_3.npy')
for i in range(y1c.shape[0]):
  if y1c[i] == 0 or y1c[i] == 1 or y1c[i] == 2:
    y1c[i] = 0
  else:
    y1c[i] = 1

for i in range(y2c.shape[0]):
  if y2c[i] == 0 or y2c[i] == 1 or y2c[i] == 2:
    y2c[i] = 0
  else:
    y2c[i] = 1

for i in range(y3c.shape[0]):
  if y3c[i] == 0 or y3c[i] == 1 or y3c[i] == 2:
    y3c[i] = 0
  else:
    y3c[i] = 1

yc = np.concatenate((y1c, y2c, y3c))


print(y1c.shape)
print(y2c.shape)
print(y3c.shape)

"""##Employing RandomForestClassifier"""

model = RandomForestClassifier()
y_pred = np.zeros(y1c.shape[0] + y2c.shape[0] + y3c.shape[0])

model.fit(np.vstack((k2, k3)), np.append(y2c, y3c))
y_pred[:k1.shape[0]] = model.predict(k1)

model.fit(np.vstack((k1, k3)), np.append(y1c, y3c))
y_pred[k1.shape[0]:k1.shape[0] + k2.shape[0]] = model.predict(k2)

model.fit(np.vstack((k1, k2)), np.append(y1c, y2c))
y_pred[k1.shape[0] + k2.shape[0]:] = model.predict(k3)

print(classification_report(yc, y_pred))

"""##Employing Conv1D.

K = 3 Fold Data Prep
"""

y_pred = np.zeros((y.shape[0]), dtype="int32")

####1st Fold
x_1st_fold = np.vstack((k2, k3))
print(x_1st_fold.shape)
y_1st_fold = np.append(y2c, y3c)
print(y_1st_fold.shape)
#y_1st_fold = tf.keras.utils.to_categorical(y_1st_fold, 1)
print(y_1st_fold.shape)

####2nd Fold
x_2nd_fold = np.vstack((k1, k3))
print(x_2nd_fold.shape)
y_2nd_fold = np.append(y1c, y3c)
print(y_2nd_fold.shape)
#y_2nd_fold = tf.keras.utils.to_categorical(y_2nd_fold, 1)
print(y_2nd_fold.shape)

###3rd Fold
x_3rd_fold = np.vstack((k1, k2))
print(x_3rd_fold.shape)
y_3rd_fold = np.append(y1c, y2c)
#y_3rd_fold = tf.keras.utils.to_categorical(y_3rd_fold, 1)
print(y_3rd_fold.shape)

"""Define Conv1D Model."""

from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def create_model():
  model = Sequential()
  model.add(Conv1D(128, 33, activation="relu", input_shape=(33, 60)))
  model.add(MaxPooling1D(2, padding = 'same'))
  model.add(Conv1D(64, 33, padding = 'same', activation="relu", input_shape=(33, 60)))
  model.add(MaxPooling1D(2, padding = 'same'))
  model.add(Conv1D(32, 33, padding = 'same', activation="relu", input_shape=(33, 60)))
  model.add(MaxPooling1D(2, padding = 'same'))
  model.add(Conv1D(16, 33, padding = 'same', activation="relu", input_shape=(33, 60)))
  model.add(MaxPooling1D(2, padding = 'same'))
  model.add(Flatten())
  model.add(Dense(1000, activation = 'relu'))
  model.add(Dropout(0.2))
  model.add(Dense(500, activation = 'relu'))
  model.add(Dropout(0.2))
  model.add(Dense(1, activation = 'sigmoid'))
  optimizer = keras.optimizers.Adam(lr=0.001)
  metrics = [
            #keras.metrics.TruePositives(name='tp'),
            #keras.metrics.FalsePositives(name='fp'),
            #keras.metrics.TrueNegatives(name='tn'),
            #keras.metrics.FalseNegatives(name='fn'),
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            f1_m
  ]
  model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = metrics)
  return model

"""Train Model."""

weights = {0:1, 1:2}
es = keras.callbacks.EarlyStopping(monitor = 'f1_m', patience = 3)
model = create_model()
model.fit(x_1st_fold, y_1st_fold, class_weight = weights, batch_size = 64, epochs = 30, callbacks = [es])
temp = model.predict(k1).reshape(-1)
y_pred[:k1.shape[0]] = temp > 0.5
#model = create_model()
model.fit(x_2nd_fold, y_2nd_fold, class_weight = weights, batch_size = 64, epochs = 30, callbacks = [es])
temp = model.predict(k2).reshape(-1)
y_pred[k1.shape[0]:k1.shape[0] + k2.shape[0]] = temp > 0.5
#model = create_model()
model.fit(x_3rd_fold, y_3rd_fold, class_weight = weights, batch_size = 64, epochs = 30, callbacks = [es])
temp = model.predict(k3).reshape(-1)
y_pred[k1.shape[0] + k2.shape[0]:] = temp > 0.5

"""Evaluate Model"""

print(classification_report(yc, y_pred))