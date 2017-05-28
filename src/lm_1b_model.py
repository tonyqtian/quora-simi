'''
Created on May 28, 2017

@author: tonyq
'''

def lm_1b_infer(args, inputLength, data_mat):
	
	BATCH_SIZE = args.train_batch_size
	NUM_TIMESTEPS = inputLength
	MAX_WORD_LEN = 50
	
# 	import tensorflow as tf
	import numpy as np
	
	from util import data_utils
	vocab = data_utils.CharsVocabulary(args.vocab_file, MAX_WORD_LEN)
	
	targets = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
	weights = np.ones([BATCH_SIZE, NUM_TIMESTEPS], np.float32)

	from util.lm_1b_eval import _LoadModel
	sess, t = _LoadModel(args.pbtxt, args.ckpt)

# 	if sentence.find('<S>') != 0:
# 		sentence = '<S> ' + sentence

	word_ids = [vocab.word_to_id(w) for w in sentence.split()]
	char_ids = [vocab.word_to_char_ids(w) for w in sentence.split()]

	inputs = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
	char_ids_inputs = np.zeros([BATCH_SIZE, NUM_TIMESTEPS, vocab.max_word_length], np.int32)
	for i in range(len(word_ids)):
		inputs[0, 0] = word_ids[i]
		char_ids_inputs[0, 0, :] = char_ids[i]

		# Add 'lstm/lstm_0/control_dependency' if you want to dump previous layer
		# LSTM.
		lstm_emb = sess.run(t['lstm/lstm_1/control_dependency'],
							feed_dict={t['char_inputs_in']: char_ids_inputs,
										t['inputs_in']: inputs,
										t['targets_in']: targets,
										t['target_weights_in']: weights})
	return lstm_emb