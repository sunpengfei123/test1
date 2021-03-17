import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import data
import numpy as np
from matplotlib import pyplot as plt

train_x = tf.convert_to_tensor(data.get_description('track1_round1_train_20210222.csv'))
train_y = tf.convert_to_tensor(data.get_label('track1_round1_train_20210222.csv'))
print(train_y.shape)

test_x, test_y = data.get_test('track1_round1_train_20210222.csv')

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(859, 100))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,dropout=0.4)))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(17, activation='sigmoid'))

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['acc']
)


# history = model.fit(train_x, train_y, batch_size=10, epochs=10,validation_data=(test_x, test_y))

print(model.summary())

# model.save("less_model.h5")
#
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model train vs validation loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper right')
# plt.show()

