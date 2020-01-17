import tensorflow as tf
from cleverhans.model import Model


class MyModel(Model):
	def __init__(self, scope, nb_classes, nb_filters, input_shape, **kwargs):
		del kwargs
		Model.__init__(self, scope, nb_classes, locals())
		self.nb_filters = nb_filters
		self.input_shape = input_shape

		# Do a dummy run of fprop to create the variables from the start
		self.fprop(tf.placeholder(tf.float32, [32] + input_shape))
		# Put a reference to the params in self so that the params get pickled
		self.params = self.get_params()

		self.d1 = tf.layers.Dense(128, activation='relu')
		self.d2 = tf.layers.Dense(2, activation='softmax')

	def fprop(self, x, **kwargs):
		del kwargs
		y = x

		with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
			y = self.d1(y)
			y = self.d2(y)
			logits = tf.reduce_mean(y, [1, 2])
			return {self.O_LOGITS: logits, self.O_PROBS: tf.nn.softmax(logits=logits)}
