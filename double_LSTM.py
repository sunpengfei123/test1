import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import data

train_x = tf.convert_to_tensor(data.get_description('track1_round1_train_20210222.csv'))
train_y = tf.convert_to_tensor(data.get_label('track1_round1_train_20210222.csv'))
print(train_x.shape)
rnn_units = 128

input_layer = keras.Input(shape=(104,1))
# x = layers.Embedding(858, 100)(input_layer)
# x = layers.Bidirectional(layers.LSTM(rnn_units,return_sequences=True, recurrent_initializer='orthogonal'))(input_layer)
# x = layers.Bidirectional(layers.LSTM(rnn_units,return_sequences=True, recurrent_initializer='orthogonal'))(x)
x = layers.LSTM(64)(input_layer)
x = layers.Dropout(rate=0.5)(x)
# x = layers.Dense(64, activation='relu')(x)
output = layers.Dense(17, activation='sigmoid')(x)

model = keras.Model(input_layer,output)
model.compile("adam", loss='binary_crossentropy',metrics=['binary_accuracy'])

history = model.fit(train_x, train_y, epochs=20, batch_size=100)

model.save("lstm_model.h5")