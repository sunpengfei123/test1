import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import data
import numpy as np

train_x = tf.convert_to_tensor(data.get_description('track1_round1_train_20210222.csv'))
train_y = tf.convert_to_tensor(data.get_label('track1_round1_train_20210222.csv'))
print(train_y.shape)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(858, 100))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(17, activation='sigmoid'))

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['acc']
)


model.fit(train_x, train_y, batch_size=5, epochs=10)

print(model.summary())

model.save("less_model.h5")

