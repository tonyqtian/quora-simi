'''
Created on Mar 24, 2017

@author: tonyq
'''
from keras.layers.core import Dense
from keras.layers.pooling import AveragePooling1D
from keras.layers.convolutional import Convolution1D
from keras.engine.topology import Layer
from keras.backend import sum, cast, floatx, mean

class DenseWithMasking(Dense):
	def __init__(self, output_dim, **kwargs):
		self.supports_masking = True
		super(DenseWithMasking, self).__init__(output_dim, **kwargs)
	
	def compute_mask(self, x, mask=None):
		return None

class AveragePooling1DWithMasking(AveragePooling1D):
	def __init__(self, pool_size, **kwargs):
		self.supports_masking = True
		super(AveragePooling1DWithMasking, self).__init__(pool_size, **kwargs)
	
	def compute_mask(self, x, mask=None):
		return None
			
class Conv1DWithMasking(Convolution1D):
	def __init__(self, **kwargs):
		self.supports_masking = True
		super(Conv1DWithMasking, self).__init__(**kwargs)
	
	def compute_mask(self, x, mask):
		return mask
	
class MeanOverTime(Layer):
	def __init__(self, mask_zero=True, **kwargs):
		self.mask_zero = mask_zero
		self.supports_masking = True
		super(MeanOverTime, self).__init__(**kwargs)

	def call(self, x, mask=None):
		if self.mask_zero:
			return sum(x, axis=1) / sum(cast(mask, floatx()), axis=1, keepdims=True)
		else:
			return mean(x, axis=1)

	def get_output_shape_for(self, input_shape):
		return (input_shape[0], input_shape[2])
	
	def compute_output_shape(self, input_shape):
		return (input_shape[0], input_shape[2])
	
	def compute_mask(self, x, mask):
		return None
	
	def get_config(self):
		config = {'mask_zero': self.mask_zero}
		base_config = super(MeanOverTime, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
	