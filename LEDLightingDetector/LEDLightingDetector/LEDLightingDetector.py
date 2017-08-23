# coding: utf-8

import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD


# データ読み込み・作成
batch_size = 32
epochs = 100

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=180,
    zoom_range=[0.8,2.0],
    width_shift_range=0.2,
    height_shift_range=0.2)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(32, 48),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    'data/validation',
    target_size=(32, 48),
    batch_size=batch_size,
    class_mode='categorical')

#print(train_generator.class_indices)


### add for TensorBoard
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

old_session = KTF.get_session()

session = tf.Session('')
KTF.set_session(session)
KTF.set_learning_phase(1)
### 


# ネットワーク定義
model = Sequential()

model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(32, 48, 3)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])


### add for TensorBoard
tb_cb = keras.callbacks.TensorBoard(log_dir="C:/Users/silverstone/tflog", histogram_freq=1, write_graph=True, write_images=True)
cbks = [tb_cb]
###


# 学習
history = model.fit_generator(
    train_generator,
    steps_per_epoch=1000,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=100,
    callbacks=cbks)
score = model.evaluate_generator(validation_generator, steps=100)
print(score)


### add for TensorBoard
KTF.set_session(old_session)
###