import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import backend as K

def customLoss(y_true, y_pred, **kwargs):
	y = tf.sign(tf.reduce_max(y_pred, axis=-1, keepdims=True) - y_pred)
	y = (y - 1) * (-1)

	return K.sum(tf.math.multiply(y,y_true))* (-1)

def customLoss2(y_true, y_pred, **kwargs):
	casted_y_true = tf.keras.backend.cast(y_true, dtype='float32')

	return K.sum(tf.math.square(y_pred) * casted_y_true) * -1


y_true = [[0., 1., 1.], [0., 1., 1.], [1., 0., 1.]]
y_pred = [[0.6, 0.2, 0.2], [0.2, 0.2, 0.6], [0.7, 0.2, 0.2]]

print(customLoss(y_true, y_pred).numpy())

y_true = [[0., 1.], [1., 0.]]
y_pred = [[0.6, 0.4], [0.8, 0.2]]

print(customLoss2(y_true, y_pred).numpy())