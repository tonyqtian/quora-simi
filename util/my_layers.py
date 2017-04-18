'''
Created on Mar 24, 2017

@author: tonyq
'''
from keras.layers.core import Dense

class DenseWithMasking(Dense):
	def __init__(self, output_dim, **kwargs):
		self.supports_masking = True
		super(DenseWithMasking, self).__init__(output_dim, **kwargs)
	
	def compute_mask(self, x, mask=None):
		return None