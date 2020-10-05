import tensorflow as tf
from inits import *


def dot(x, y, sparse=False):
	if sparse:
		res = tf.sparse_tensor_dense_matmul(x, y)
	else:
		res = tf.matmul(x, y)
	return res

class Dense():
	def __init__(self, name, input_dim, output_dim, placeholders, act,use_dropout = True, support_id = 0, sparse_inputs = False, bias = False, concat = False):
		self.act = act
		self.sparse_inputs = sparse_inputs
		self.bias = bias
		self.debug = None
		self.use_dropout = use_dropout
		if self.use_dropout:
			self.dropout = placeholders['dropout']
		self.concat = concat
		self.name = name
		self.bias = bias
		if support_id is None:
			self.support = placeholders['support_0']
		else:
			self.support = placeholders['support_' + str(support_id)]
		self.vars = {}
		with tf.variable_scope(self.name + '_vars'):
			#self.vars['weights'] = uniform([input_dim, output_dim], 0.5, name='weights')
			self.vars['weights'] = normal([input_dim, output_dim], name='weights')
			if self.bias:
				self.vars['bias'] = zeros([output_dim], name='bias')
	def __call__(self, inputs):
		x = inputs
		# consider non sparse
		if self.use_dropout:
			#if tf.equal(self.dropout, tf.constant(0.5, dtype=tf.float32)):
			x = tf.nn.dropout(x, 1 - self.dropout)
			#else:
			#	x = self.dropout * x
		#print(x)
		#self.debug = self.vars['weights']
		output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)
		if self.bias:
			output += self.vars['bias']
		if self.act == None:
			return output
		return self.act(output)

class GraphConvolution1():
	def __init__(self, name, input_dim, output_dim, placeholders, use_dropout=False,
		support = None, sparse_inputs=False, act = tf.nn.relu, bias=True):
		self.act = act
		self.sparse_inputs = sparse_inputs

		self.debug = None
		self.placeholders = placeholders
		self.dropout = placeholders['dropout']
		self.use_dropout = use_dropout
		if self.use_dropout:
			self.dropout = placeholders['dropout']
		#else:
		#	self.dropout = tf.constant(0.0)
		self.name = name
		self.bias = bias
		if support is None:
			#print(placeholders)
			self.support = placeholders['support']
		else:
			self.support = support
		self.vars={}
		with tf.variable_scope(self.name +'_vars'):
			#self.vars['combine'] = uniform([len(self.support)], 0.01, name = 'combine')
			#self.vars['weights'] = uniform([input_dim, output_dim], 0.01, name ='weights')
			for i in range(len(self.support)):
			#	self.vars['weights_'+str(i)] = 
				#self.vars['weights_'+str(i)] = uniform([input_dim, output_dim], 0.5, name='weights_'+str(i))
				self.vars['weights_'+str(i)] = normal([input_dim, output_dim], name='weights_' + str(i))
				#print(self.vars['weights_'+str(i)])
			if self.bias:
				self.vars['bias'] = zeros([output_dim], name='bias')

	def __call__(self, inputs):
		x = inputs
		if self.use_dropout:
			print("using DROPOUT")
			#if self.dropout < tf.constant(1, dtype=tf.float32):
			#if tf.equal(self.dropout, tf.constant(0.5, dtype=tf.float32)):
			x = tf.nn.dropout(x, 1 - self.dropout)
		else:
			print("NOT using dropout")
			#	x = self.dropout * x
		#output = 0
		outputs=[]
		#for i in range(len(self.support)):
		#	pre_sup = dot(x, self.vars['weights'], sparse=self.sparse_inputs)
		#	output += self.vars['combine'][i]* dot(self.support[i], pre_sup, sparse=True)
	
		for i in range(len(self.support)):
			pre_sup = dot(x, self.vars['weights_'+str(i)], sparse=self.sparse_inputs)
			output = dot(self.support[i], pre_sup, sparse=True)
			outputs.append(output)
			#output += pre_sup
		outputs=tf.concat(outputs,axis=-1)
		if self.bias:
			output+=self.vars['bias']
		return self.act(output)


class OneHot():
	def __init__(self, name, input_dim, output_dim, placeholders, sparse_inputs=True):
		self.sparse_inputs = sparse_inputs
		self.name = name
		self.vars={}
		with tf.variable_scope(self.name +"_vars"):
			self.vars['weights'] = normal([input_dim, output_dim], name="weights")
	def __call__(self, inputs):
		x=inputs
		output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)
		return output
			


class GraphConvolution():
	def __init__(self, name, input_dim, output_dim, placeholders, use_dropout=False,
		support_id = None, sparse_inputs=False, act = tf.nn.relu, bias=True, concat = True):
		self.act = act
		self.sparse_inputs = sparse_inputs
		self.concat = concat

		self.debug = None
		self.placeholders = placeholders
		self.dropout = placeholders['dropout']
		self.use_dropout = use_dropout
		if self.use_dropout:
			self.dropout = placeholders['dropout']
		#else:
		#	self.dropout = tf.constant(0.0)
		self.name = name
		self.bias = bias
		if support_id is None:
			#print(placeholders)
			self.support = placeholders['support_0']
		else:
			self.support = placeholders['support_' + str(support_id)]
		self.vars={}
		with tf.variable_scope(self.name +'_vars'):
			for i in range(len(self.support)):
			#	self.vars['weights_'+str(i)] = 
				#self.vars['weights_'+str(i)] = uniform([input_dim, output_dim], 0.5, name='weights_'+str(i))
				if concat:
					tmp = int(output_dim/(1.0 * len(self.support)))
				else:
					tmp = output_dim
				self.vars['weights_'+str(i)] = normal([input_dim, tmp], name='weights_' + str(i))
				#print(self.vars['weights_'+str(i)])
			if self.bias:
				self.vars['bias'] = zeros([output_dim], name='bias')

	def __call__(self, inputs):
		x = inputs
		if self.use_dropout:
			#if self.dropout < tf.constant(1, dtype=tf.float32):
			#if tf.equal(self.dropout, tf.constant(0.5, dtype=tf.float32)):
			x = tf.nn.dropout(x, 1 - self.dropout)
			print("Using dropout")
			#else:
			#	x = self.dropout * x
		else:
			print("NOT using droput")
		outputs=[]
		#for i in range(len(self.support)):
		#	pre_sup = dot(x, self.vars['weights'], sparse=self.sparse_inputs)
		#	output += self.vars['combine'][i]* dot(self.support[i], pre_sup, sparse=True)
		#order_rep = dot(self.F, self.vars['order_weights'])
	
		for i in range(len(self.support)):
			pre_sup = dot(x, self.vars['weights_'+str(i)], sparse=self.sparse_inputs)
			output = dot(self.support[i], pre_sup, sparse=True)
			outputs.append(output)
		if self.concat:
			outputs=tf.concat(outputs,axis=-1)
		else:
			outputs=tf.add_n(outputs)/(1.0 * len(self.support))
		
		if self.bias:
			outputs+=self.vars['bias']
		return self.act(outputs)






