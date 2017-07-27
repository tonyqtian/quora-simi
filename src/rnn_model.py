'''
Created on Mar 18, 2017

@author: tonyq
'''
import logging
# from random import random
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dropout
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.merge import Concatenate
from keras.engine.topology import Input
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Convolution1D
from util.my_layers import DenseWithMasking
from util.my_layers import MeanOverTime
from util.my_layers import Conv1DWithMasking, MaxOverTime

from tensorflow import device

logger = logging.getLogger(__name__)

mem_opt_dict = {'cpu':0, 'mem':1, 'gpu': 2}


def getModel(args, input_length, vocab_size, embd, feature_length=0):
	rnn_opt = mem_opt_dict[args.rnn_opt]
	rnn_dim = args.rnn_dim
	rnn_dropout = args.rnn_dropout
	dense_dropout = args.dense_dropout
# 	if args.activation == 'sigmoid':
# 		final_init = 'he_normal'
# 	else:
# 		final_init = 'he_uniform'
	with device('/cpu:0'):
		if type(embd) is type(None):
			embd_layer = Embedding(vocab_size, args.embd_dim, mask_zero=args.use_mask, trainable=True)
		else:
			embd_layer = Embedding(vocab_size, args.embd_dim, mask_zero=args.use_mask, weights=[embd], trainable=False)
		
	if args.bidirectional:
		rnn_layer = Bidirectional(LSTM(rnn_dim, return_sequences=False, implementation=rnn_opt, 
										dropout=rnn_dropout, recurrent_dropout=rnn_dropout))
	else:
		rnn_layer = LSTM(rnn_dim, return_sequences=False, implementation=rnn_opt, 
						dropout=rnn_dropout, recurrent_dropout=rnn_dropout)

	if args.mot_layer:
		if args.use_mask:
			w2v_dense_layer = TimeDistributed(DenseWithMasking(args.embd_dim * 2 // 3))
		else:
			w2v_dense_layer = TimeDistributed(Dense(args.embd_dim*2//3))
		meanpool = MeanOverTime()

	if args.cnn_dim:
		if args.use_mask:
			conv1dw2 = Conv1DWithMasking(filters=args.cnn_dim, kernel_size=2, padding='valid', strides=1)
			conv1dw3 = Conv1DWithMasking(filters=args.cnn_dim, kernel_size=3, padding='valid', strides=1)
		else:
			conv1dw2 = Convolution1D(filters=args.cnn_dim, kernel_size=2, padding='valid', strides=1)
			conv1dw3 = Convolution1D(filters=args.cnn_dim, kernel_size=3, padding='valid', strides=1)
		maxpool = MaxOverTime()
# 		maxpool = MaxPooling1DWithMasking(pool_size=input_length-1, padding='valid')
	
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
	merged = []
	
	if args.mot_layer:
		vec1_w2v = w2v_dense_layer(vec1)
		vec2_w2v = w2v_dense_layer(vec2)
		vec1_w2v = meanpool(vec1_w2v)
		vec2_w2v = meanpool(vec2_w2v)
		merged += [vec1_w2v, vec2_w2v]
		
	if args.rnn_layer:
		vec1_rnn = rnn_layer(vec1)
		vec2_rnn = rnn_layer(vec2)
		merged += [vec1_rnn, vec2_rnn]
	# Conv Layer
	if args.cnn_dim:
		vec1_cnnw2 = conv1dw2(vec1)
		vec2_cnnw2 = conv1dw2(vec2)
		vec1_cnnw3 = conv1dw3(vec1)
		vec2_cnnw3 = conv1dw3(vec2)
		vec1_cnnw2 = maxpool(vec1_cnnw2)
		vec2_cnnw2 = maxpool(vec2_cnnw2)
		vec1_cnnw3 = maxpool(vec1_cnnw3)
		vec2_cnnw3 = maxpool(vec2_cnnw3)
		merged += [vec1_cnnw2, vec2_cnnw2, vec1_cnnw3, vec2_cnnw3]
	
	if feature_length:
		feature_input = Input(shape=(feature_length,), dtype='float32')
		if args.use_mask:
			featured = DenseWithMasking(args.dense_dim//2, kernel_initializer='he_uniform',
								activation='relu')(feature_input)
		else:
			featured = Dense(args.dense_dim//2, kernel_initializer='he_uniform',
								activation='relu')(feature_input)
		merged += [featured]

	merged = Concatenate()(merged)
	merged = BatchNormalization()(merged)
	merged = Dropout(dense_dropout)(merged)

	if args.use_mask:
		merged = DenseWithMasking(args.dense_dim, kernel_initializer='he_uniform', activation='relu')(merged)
	else:
		merged = Dense(args.dense_dim, kernel_initializer='he_uniform', activation='relu')(merged)
	merged = BatchNormalization()(merged)
	merged = Dropout(dense_dropout)(merged)

	if args.use_mask:
		preds = DenseWithMasking(1, kernel_initializer='he_normal', activation='sigmoid')(merged)
	else:
		preds = Dense(1, kernel_initializer='he_normal', activation='sigmoid')(merged)

	if feature_length:
		model = Model(inputs=[sequence_1_input, sequence_2_input, feature_input], outputs=preds)
	else:
		model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)
	return model
