'''
Created on Mar 18, 2017

@author: tonyq
'''
import logging
from random import random
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dropout
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import Concatenate
from keras.engine.topology import Input
from keras.models import Model

from keras.layers.normalization import BatchNormalization
# from keras.layers import Dense
from util.my_layers import DenseWithMasking
from util.my_layers import MeanOverTime

logger = logging.getLogger(__name__)

mem_opt_dict = {'cpu':0, 'mem':1, 'gpu': 2}

def getModel(args, input_length, vocab_size, embd, feature_length=0):
	rnn_opt = mem_opt_dict[args.rnn_opt]
	rnn_dim = args.rnn_dim
	dropout_prob = args.dropout_prob + random() * 0.15
# 	if args.activation == 'sigmoid':
# 		final_init = 'he_normal'
# 	else:
# 		final_init = 'he_uniform'
		
	if type(embd) is type(None):
		embd_layer = Embedding(vocab_size, args.embd_dim, mask_zero=True, trainable=True)
	else:
		embd_layer = Embedding(vocab_size, args.embd_dim, mask_zero=True, weights=[embd], trainable=False)
		
	if args.bidirectional:
		rnn_layer = Bidirectional(LSTM(rnn_dim, return_sequences=False, implementation=rnn_opt, 
										dropout=dropout_prob, recurrent_dropout=dropout_prob))
	else:
		rnn_layer = LSTM(rnn_dim, return_sequences=False, implementation=rnn_opt, 
						dropout=dropout_prob, recurrent_dropout=dropout_prob)
# 	# hidden rnn layer
# 	from keras.models import Sequential
# 	rnn_model = Sequential()
# 	for _ in range(args.rnn_layer-1):
# 		if args.bidirectional:
# 			rnn_model.add(Bidirectional(LSTM(rnn_dim, return_sequences=True, implementation=rnn_opt, 
# 											dropout=dropout_prob, recurrent_dropout=dropout_prob)))
# 		else:
# 			rnn_model.add(LSTM(rnn_dim, return_sequences=True, implementation=rnn_opt, 
# 							dropout=dropout_prob, recurrent_dropout=dropout_prob))
# 
# 	# output rnn layer	
# 	if args.attention:
# 		from util.attention_wrapper import Attention		
# 		rnn_model.add(Attention(LSTM(rnn_dim, return_sequences=False, implementation=rnn_opt, 
# 									dropout=dropout_prob, recurrent_dropout=dropout_prob)))
# 	else:
# 		if args.bidirectional:
# 			rnn_model.add(Bidirectional(LSTM(rnn_dim, return_sequences=False, implementation=rnn_opt,
# 											 dropout=dropout_prob, recurrent_dropout=dropout_prob)))
# 		else:
# 			rnn_model.add(LSTM(rnn_dim, return_sequences=False, implementation=rnn_opt, 
# 							dropout=dropout_prob, recurrent_dropout=dropout_prob))
	
	sequence_1_input = Input(shape=(input_length,), dtype='int32')
	sequence_2_input = Input(shape=(input_length,), dtype='int32')
	vec1 = embd_layer(sequence_1_input)
	vec2 = embd_layer(sequence_2_input)
	vec1_w2v = MeanOverTime()(vec1)
	vec2_w2v = MeanOverTime()(vec2)
	vec1_rnn = rnn_layer(vec1)
	vec2_rnn = rnn_layer(vec2)
	
	if feature_length:
		feature_input = Input(shape=(feature_length,), dtype='float32')
		featured = DenseWithMasking(feature_length*3//5, kernel_initializer='he_uniform', 
								activation='relu')(feature_input)
		merged = Concatenate()([vec1_rnn, vec1_w2v, vec2_rnn, vec2_w2v, featured])
	else:	
		merged = Concatenate()([vec1_rnn, vec1_w2v, vec2_rnn, vec2_w2v])
	merged = BatchNormalization()(merged)
	merged = Dropout(dropout_prob)(merged)

	merged = DenseWithMasking(rnn_dim, kernel_initializer='he_uniform', activation='relu')(merged)
	merged = BatchNormalization()(merged)
	merged = Dropout(dropout_prob)(merged)
	
	preds = DenseWithMasking(1, kernel_initializer='he_normal', activation='sigmoid')(merged)
	if feature_length:
		model = Model(inputs=[sequence_1_input, sequence_2_input, feature_input], outputs=preds)
	else:
		model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)
	return model
