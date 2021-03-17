import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

query = tf.convert_to_tensor(np.asarray([[[1., 1., 1., 3.]]]))

key_list = tf.convert_to_tensor(np.asarray([[[1., 1., 2., 4.], [4., 1., 1., 3.], [1., 1., 2., 1.]],
                                            [[1., 0., 2., 1.], [1., 2., 1., 2.], [1., 0., 2., 1.]]]))

query_value_attention_seq = tf.keras.layers.Attention()([key_list, key_list])

# print(query.shape)
# print(query)
print(key_list)
print(query_value_attention_seq)


# scores = tf.matmul(query, key_list, transpose_b=True)
#
# print(scores)