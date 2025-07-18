# %tensorflow_version 2.x

import time
import numpy as np
import tensorflow as tf

tf.__version__

(X_train,y_train),(X_test,y_test)=tf.keras.datasets.mnist.load_data()

X_train=X_train/255.
X_test=X_test/255.

X_train.shape

X_train=X_train.reshape(-1,28*28)
X_test=X_test.reshape(-1,28*28)

X_train.shape

"""Normal model"""

model_normal=tf.keras.models.Sequential()

model_normal.add(tf.keras.layers.Dense(units=128,activation='relu',input_shape=(784,)))

model_normal.add(tf.keras.layers.Dropout(rate=0.2))

model_normal.add(tf.keras.layers.Dense(units=10,activation='softmax'))

model_normal.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])

"""Distibuted Strategy"""

distribute=tf.distribute.MirroredStrategy()

with distribute.scope():
  model_distributed=tf.keras.models.Sequential()
  model_distributed.add(tf.keras.layers.Dense(units=128,activation='relu',input_shape=(784,)))
  model_distributed.add(tf.keras.layers.Dropout(rate=0.2))
  model_distributed.add(tf.keras.layers.Dense(units=10,activation='softmax'))
  model_distributed.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])

