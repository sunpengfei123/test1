import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import data
from matplotlib import pyplot as plt

train_x = tf.convert_to_tensor(data.get_description('track1_round1_train_20210222.csv'))
train_y = tf.convert_to_tensor(data.get_label('track1_round1_train_20210222.csv'))
print(train_x.shape)

test_x, test_y = data.get_test('track1_round1_train_20210222.csv')

rnn_units = 32

input_layer = keras.Input(shape=(104,1))
# x = layers.Embedding(input_dim=859, output_dim=10)(input_layer)
x = layers.Conv1D(filters=rnn_units,kernel_size=4,padding='valid')(input_layer)
x = layers.Conv1D(filters=rnn_units,kernel_size=4,padding='valid')(x)
x = layers.AveragePooling1D()(x)
x = layers.Conv1D(filters=rnn_units/2,kernel_size=4,padding='valid')(x)
# x = layers.AveragePooling1D()(x)
x = layers.Conv1D(filters=rnn_units/4,kernel_size=3,padding='valid')(x)
x = layers.AveragePooling1D()(x)
# x = layers.Dropout(rate=0.5)(x)
x = layers.Conv1D(filters=rnn_units/4,kernel_size=3,padding='valid')(x)
x = layers.Conv1D(filters=rnn_units/4,kernel_size=3,padding='valid')(x)
x = layers.AveragePooling1D()(x)
x = layers.Dropout(rate=0.5)(x)
x = layers.Flatten()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(256, activation='relu')(x)
# x = layers.Dropout(rate=0.5)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(rate=0.5)(x)
output = layers.Dense(17, activation='sigmoid')(x)

model = keras.Model(input_layer,output)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=keras.losses.BinaryCrossentropy(),metrics=['binary_accuracy'])

history = model.fit(train_x, train_y, epochs=100, batch_size=20, validation_data=(test_x, test_y), shuffle=False)
model.save("model_conv_310.h5")

print(model.summary())

# plot train and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()