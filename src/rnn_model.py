'''
Created on Mar 18, 2017

@author: tonyq
'''
import logging
from keras.models import Sequential
# from keras.layers import Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dropout
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import Concatenate
from keras.engine.topology import Input
from keras.models import Model

from util.my_layers import DenseWithMasking
from keras.layers.normalization import BatchNormalization
from random import random

logger = logging.getLogger(__name__)

mem_opt_dict = {'cpu':0, 'mem':1, 'gpu': 2}

def getModel(args, input_length, vocab_size, embd, embd_trainable=True, feature_length=0):
	rnn_opt = mem_opt_dict[args.rnn_opt]
	rnn_dim = args.rnn_dim
	dropout_prob = args.dropout_prob + random() * 0.15
# 	if args.activation == 'sigmoid':
# 		final_init = 'he_normal'
# 	else:
# 		final_init = 'he_uniform'
	
	rnn_model = Sequential()
	if type(embd) is type(None):
		rnn_model.add(Embedding(vocab_size, args.embd_dim, mask_zero=True, trainable=embd_trainable))
	else:
		rnn_model.add(Embedding(vocab_size, args.embd_dim, mask_zero=True, weights=[embd], 
							trainable=embd_trainable))
	# hidden rnn layer
	for _ in range(args.rnn_layer-1):
		if args.bidirectional:
			rnn_model.add(Bidirectional(LSTM(rnn_dim, return_sequences=True, implementation=rnn_opt, 
											dropout=dropout_prob, recurrent_dropout=dropout_prob)))
		else:
			rnn_model.add(LSTM(rnn_dim, return_sequences=True, implementation=rnn_opt, 
							dropout=dropout_prob, recurrent_dropout=dropout_prob))

	# output rnn layer	
	if args.attention:
		from util.attention_wrapper import Attention		
		rnn_model.add(Attention(LSTM(rnn_dim, return_sequences=False, implementation=rnn_opt, 
									dropout=dropout_prob, recurrent_dropout=dropout_prob)))
	else:
		if args.bidirectional:
			rnn_model.add(Bidirectional(LSTM(rnn_dim, return_sequences=False, implementation=rnn_opt,
											 dropout=dropout_prob, recurrent_dropout=dropout_prob)))
		else:
			rnn_model.add(LSTM(rnn_dim, return_sequences=False, implementation=rnn_opt, 
							dropout=dropout_prob, recurrent_dropout=dropout_prob))
	
	sequence_1_input = Input(shape=(input_length,), dtype='int32')
	sequence_2_input = Input(shape=(input_length,), dtype='int32')
	vec1 = rnn_model(sequence_1_input)
	vec2 = rnn_model(sequence_2_input)
	
	if feature_length:
		feature_input = Input(shape=(feature_length,), dtype='float32')
		featured = BatchNormalization()(feature_input)
		featured = DenseWithMasking(feature_length*3//5, kernel_initializer='he_uniform', activation='relu')(featured)
# 		featured = DenseWithMasking(feature_length//2, activation='relu')(featured)
		merged = Concatenate()([vec1, vec2, featured])
	else:	
		merged = Concatenate()([vec1, vec2])
	merged = BatchNormalization()(merged)
	merged = Dropout(dropout_prob)(merged)

	merged = DenseWithMasking(rnn_dim, kernel_initializer='he_uniform', activation='relu')(merged)
# 	merged = DenseWithMasking(rnn_dim, activation='relu')(merged)
	merged = BatchNormalization()(merged)
	merged = Dropout(dropout_prob)(merged)
	
	preds = DenseWithMasking(1, kernel_initializer='he_normal', activation='sigmoid')(merged)
# 	preds = DenseWithMasking(1, activation='sigmoid')(merged)
	if feature_length:
		model = Model(inputs=[sequence_1_input, sequence_2_input, feature_input], outputs=preds)
	else:
		model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)
	return model
