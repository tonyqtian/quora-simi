'''
Created on Mar 18, 2017

@author: tonyq
'''
import logging
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Activation, Dropout
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import Concatenate
from keras.engine.topology import Input
from keras.models import Model

from util.attention_wrapper import Attention
from util.my_layers import DenseWithMasking

logger = logging.getLogger(__name__)

mem_opt_dict = {'cpu':0, 'mem':1, 'gpu': 2}

def getModel(args, input_length, vocab_size, embd, embd_trainable=True):
	embd_dim = args.embd_dim
	rnn_opt = mem_opt_dict[args.rnn_opt]
	rnn_dim = args.rnn_dim
# 	dropout_W = args.dropout_w
# 	dropout_U = args.dropout_u
	if args.activation == 'sigmoid':
		final_init = 'he_normal'
	else:
		final_init = 'he_uniform'
	
	rnn_model = Sequential()
	if type(embd) is type(None):
		rnn_model.add(Embedding(vocab_size, embd_dim, mask_zero=True, trainable=embd_trainable))
	else:
		rnn_model.add(Embedding(vocab_size, embd_dim, mask_zero=True, weights=[embd], trainable=embd_trainable))
	# hidden rnn layer
	for _ in range(args.rnn_layer-1):
		if args.bidirectional:
			rnn_model.add(Bidirectional(LSTM(rnn_dim, return_sequences=True, implementation=rnn_opt, dropout=args.dropout_prob)))
		else:
			rnn_model.add(LSTM(rnn_dim, return_sequences=True, implementation=rnn_opt, dropout=args.dropout_prob))
# 		rnn_model.add(Dropout(args.dropout_prob))
	# output rnn layer	
	if args.attention:			
		rnn_model.add(Attention(LSTM(rnn_dim, return_sequences=False, implementation=rnn_opt, dropout=args.dropout_prob)))
	else:
		if args.bidirectional:
			rnn_model.add(Bidirectional(LSTM(rnn_dim, return_sequences=False, implementation=rnn_opt, dropout=args.dropout_prob)))
		else:
			rnn_model.add(LSTM(rnn_dim, return_sequences=False, implementation=rnn_opt, dropout=args.dropout_prob))
# 	rnn_model.add(Dropout(args.dropout_prob))
	
	sequence_1_input = Input(shape=(input_length,), dtype='int32')
	sequence_2_input = Input(shape=(input_length,), dtype='int32')
	vec1 = rnn_model(sequence_1_input)
	vec2 = rnn_model(sequence_2_input)

	merged = Concatenate()([vec1, vec2])
# 	merged = merge([vec1, vec2], mode='concat')
	merged = Dropout(args.dropout_prob)(merged)
# 	merged = BatchNormalization()(merged)	
# 	merged = Dense(num_dense, activation=act)(merged)
# 	merged = Dropout(rate_drop_dense)(merged)
# 	merged = BatchNormalization()(merged)
	
	merged = DenseWithMasking(1, kernel_initializer=final_init)(merged)
	preds = Activation(args.activation)(merged)

	model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)
	return model
