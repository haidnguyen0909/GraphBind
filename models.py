import numpy as np   
import tensorflow as tf   
from layers import *



#define metric


def masked_rmse(preds, labels, mask):
	mask = tf.cast(mask, dtype=tf.float32)
	#mask = tf.reshape(mask, [len(mask),1])
	mask = tf.reshape(mask,[-1])
	mask/=tf.reduce_mean(mask)

	loss = tf.squared_difference(preds, labels)
	loss = tf.reshape(loss,[-1])
	loss *= mask
	loss = tf.reduce_mean(loss)
	return loss

def masked_sigmoid_cross_entropy(pred, y, mask):
	pred = tf.reshape(pred, [-1])
	y = tf.reshape(y, [-1])
	loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=pred)
	mask = tf.cast(mask, dtype=tf.float32)
	mask = tf.reshape(mask,[-1])
	mask/=tf.reduce_mean(mask)
	loss*=mask
	return tf.reduce_mean(loss)

def masked_accuracy_sigmoid(preds, labels, mask):
	preds = tf.reshape(preds, [-1])
	labels = tf.reshape(labels, [-1])
	correct = tf.equal(tf.to_float(preds>=0.5), tf.to_float(labels>=0.5))
	acc_all = tf.cast(correct, tf.float32)
	mask = tf.cast(mask, dtype=tf.float32)
	mask = tf.reshape(mask,[-1])
	mask/= tf.reduce_mean(mask)
	acc_all*=mask
	return tf.reduce_mean(acc_all)



def masked_softmax_cross_entropy(preds, labels, mask):
	
	loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=preds, labels = labels)
	mask = tf.cast(mask, dtype=tf.float32)
	mask = tf.reshape(mask,[-1])
	mask/=tf.reduce_mean(mask)
	loss*=mask
	return tf.reduce_mean(loss)

def masked_accuracy(preds, labels, mask):
	correct = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
	acc_all = tf.cast(correct, tf.float32)
	mask = tf.cast(mask, dtype=tf.float32)
	mask = tf.reshape(mask,[-1])
	mask/= tf.reduce_mean(mask)
	acc_all*=mask
	return tf.reduce_mean(acc_all)

#class dGCN():
#	def __init__(self, name, placeholders, input_dim, options):
class dGCN():
	def __init__(self, name, placeholders, input_dim, options):
		self.input_dim = input_dim
		self.output_dim = placeholders['labels'].get_shape().as_list()[1]
		self.placeholders = placeholders
		lr = options['learning_rate']
		self.options = options
		self.name = name

		self.layers_0=[]
		self.layers_1=[]

		self.vars_0 = {}
		self.vars_1 = {}
		self.activations_0 = []
		self.activations_1 = []
		self.vars={}

		self.inputs_0 = placeholders['features_0']
		self.inputs_1 = placeholders['features_1']
		self.outputs = None

		self.loss = 0
		self.accuracy = 0
		with tf.variable_scope(self.name + '_vars'):
			self.vars['combine'] = uniform([2, 1], name='combine')

		self.optimizer = tf.train.AdamOptimizer(learning_rate = lr)
		self.opt_op = None
		self.build()



	def _build(self):
		self.layers_0.append(GraphConvolution(
			'GC00', 
			input_dim = self.input_dim,
			output_dim = self.options['hidden1'],
			placeholders = self.placeholders,
			act = tf.nn.relu,
			use_dropout=False,
			support_id = 0,
			sparse_inputs = True,
			bias = True,
			concat = False
			))
		self.layers_0.append(GraphConvolution(
			'GC01',
			input_dim = self.options['hidden1'],
			output_dim = self.output_dim,
			placeholders = self.placeholders,
			act = lambda x:x,
			#act = tf.nn.relu,
			use_dropout=True,
			support_id = 0,
			sparse_inputs = False,
			bias=True,
			concat = False
			))

		self.layers_1.append(GraphConvolution(
			'GC10', 
			input_dim = self.input_dim,
			output_dim = self.options['hidden1'],
			placeholders = self.placeholders,
			act = tf.nn.relu,
			use_dropout=False,
			support_id = 1,
			sparse_inputs = True,
			bias = True,
			concat = True
			))
		self.layers_1.append(GraphConvolution(
			'GC11',
			input_dim = self.options['hidden1'],
			output_dim = self.output_dim,
			placeholders = self.placeholders,
			act = lambda x:x,
			use_dropout=True,
			support_id = 1,
			sparse_inputs = False,
			bias=True,
			concat = False
			))
		
	def build(self):
		with tf.variable_scope(self.name):
			self._build()
		self.activations_0.append(self.inputs_0)
		self.activations_1.append(self.inputs_1)

		for layer in self.layers_0:
			#print("this is layer: %s" % layer.name)
			hidden = layer(self.activations_0[-1])
			self.activations_0.append(hidden)
		for layer in self.layers_1:
			#print("this is layer: %s" % layer.name)
			hidden = layer(self.activations_1[-1])
			self.activations_1.append(hidden)
		
		tmp = tf.concat([self.activations_0[-1],self.activations_1[-1]], axis=-1)
		self.debug = [self.activations_0[-2], self.placeholders['labels'], self.placeholders['labels_mask'], self.activations_0[-1]]
		self.outputs = self.activations_1[-1]
		#self.outputs = tf.reduce_mean(tmp, axis=1)
		#self.outputs = dot(tmp, self.vars['combine'])
		variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
		self.vars = {var.name:var for var in variables},

		# build metrics
		self._loss()
		self._accuracy()
		self.opt_op = self.optimizer.minimize(self.loss)

	def predict(self):
		return self.outputs
	def _loss(self):
		self.loss = masked_sigmoid_cross_entropy(self.outputs, self.placeholders['labels'], self.placeholders['labels_mask'])
		for layer in self.layers_0:
			for var in layer.vars.values():
				self.loss += self.options['weight_decay'] * tf.nn.l2_loss(var)
		for layer in self.layers_1:
			for var in layer.vars.values():
				self.loss += self.options['weight_decay'] * tf.nn.l2_loss(var)

	def _accuracy(self):
		#self.accuracy = masked_rmse(self.outputs, self.placeholders['labels'], self.placeholders['labels_mask'])

		self.accuracy = masked_accuracy_sigmoid(tf.nn.sigmoid(self.outputs), self.placeholders['labels'], self.placeholders['labels_mask'])
		self.predloss = masked_sigmoid_cross_entropy(self.outputs, self.placeholders['labels'], self.placeholders['labels_mask'])
		#self.debug = self.outputs
		self.preds = self.outputs
		self.labels = self.placeholders['labels']


class GCN():
	def __init__(self, name, placeholders, input_dim, options):
		self.input_dim = input_dim
		self.output_dim = placeholders['labels'].get_shape().as_list()[1]
		self.placeholders = placeholders
		lr = options['learning_rate']
		
		self.options = options
		self.name = name


		#print(self.placeholders)
		#self.debug = None
		
		# inti
		self.vars = {}
		self.placeholders = placeholders
		self.layers=[]
		self.activations = []

		self.inputs = placeholders['features']
		self.outputs = None
		self.loss = 0


		self.loss = 0
		self.accuracy = 0

		self.optimizer = tf.train.AdamOptimizer(learning_rate = lr)
		self.opt_op = None

		self.build()
		

	def _build(self):

		#self.layers.append(OneHot(
		#	'Onehot', 
		#	input_dim = self.input_dim,
		#	output_dim = self.options['hidden1'],
		#	placeholders = self.placeholders,
		#	sparse_inputs = True,
		#	))
		self.layers.append(GraphConvolution(
			'GC1', 
			input_dim = self.input_dim,
			output_dim = self.options['hidden1'],
			#output_dim = self.output_dim,
			placeholders = self.placeholders,
			act = tf.nn.relu,
			#act = lambda x:x,
			use_dropout=False,
			sparse_inputs = True,
			bias = True,
			concat = True
			))
		#tmp = self.layers[-1]
		#self.debug = tmp.vars['weights_0']

		#self.debug.append(self.layers[-1].debug)

		self.layers.append(GraphConvolution(
			'GC2',
			input_dim = self.options['hidden1'],
			output_dim = self.output_dim,
			placeholders = self.placeholders,
			#act = tf.nn.relu,
			#act = tf.nn.sigmoid,
			act = lambda x:x,
			use_dropout=True,
			sparse_inputs = False,
			bias=True,
			concat = False
			))
		#self.layers.append(Average(
		#	'AVG', 
		#	input_dim=self.output_dim, 
		##	placeholders = self.placeholders
		#	))


	def build(self):
		with tf.variable_scope(self.name):
			self._build()
		self.activations.append(self.inputs)
		for layer in self.layers:
			#print("this is layer: %s" % layer.name)
			hidden = layer(self.activations[-1])
			self.activations.append(hidden)
		

		self.outputs = self.activations[-1]
		variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
		self.vars = {var.name:var for var in variables}

		# build metrics
		self._loss()
		self._accuracy()
		self.opt_op = self.optimizer.minimize(self.loss)

		#self.preds = tf.nn.softmax(self.outputs)
		#self.debug = self.layers[-1].vars['weights_0']
		

	def predict(self):
		return self.outputs
	def _loss(self):
		self.loss = masked_sigmoid_cross_entropy(self.outputs, self.placeholders['labels'], self.placeholders['labels_mask'])
		for layer in self.layers_0:
			for var in layer.vars.values():
				self.loss += self.options['weight_decay'] * tf.nn.l2_loss(var)
		for layer in self.layers_1:
			for var in layer.vars.values():
				self.loss += self.options['weight_decay'] * tf.nn.l2_loss(var)

	def _accuracy(self):
		#self.accuracy = masked_rmse(self.outputs, self.placeholders['labels'], self.placeholders['labels_mask'])

		self.accuracy = masked_accuracy_sigmoid(tf.nn.sigmoid(self.outputs), self.placeholders['labels'], self.placeholders['labels_mask'])
		self.predloss = masked_sigmoid_cross_entropy(self.outputs, self.placeholders['labels'], self.placeholders['labels_mask'])
		self.debug = self.outputs
		self.preds = self.outputs
		self.labels = self.placeholders['labels']
		






