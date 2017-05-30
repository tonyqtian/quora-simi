'''
Created on May 28, 2017

@author: tonyq
'''
import sys
from tqdm._tqdm import tqdm

def lm_1b_infer(args, inputLength, data_mat):
	
	BATCH_SIZE = args.train_batch_size
	NUM_TIMESTEPS = inputLength
	MAX_WORD_LEN = 50

	import numpy as np
	
	from util import data_utils
	vocab = data_utils.CharsVocabulary(args.vocab_file, MAX_WORD_LEN)
	
	targets = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
	weights = np.ones([BATCH_SIZE, NUM_TIMESTEPS], np.float32)

	from util.lm_1b_eval import _LoadModel
	sess, t = _LoadModel(args.pbtxt, args.ckpt)

	inputs = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
	char_ids_inputs = np.zeros([BATCH_SIZE, NUM_TIMESTEPS, vocab.max_word_length], np.int32)
	final_vec = np.array([])
	idx = 0
	
	for line in tqdm(data_mat, file=sys.stdout):
		word_ids = [vocab.word_to_id(w) for w in line]
		char_ids = [vocab.word_to_char_ids(w) for w in line]
		if len(word_ids) <= NUM_TIMESTEPS:
			inputs[idx, -len(word_ids):] = word_ids
			char_ids_inputs[idx, -len(char_ids):, :] = char_ids
		else:
			inputs[idx, :] = word_ids[:NUM_TIMESTEPS]
			char_ids_inputs[idx, :, :] = char_ids[:NUM_TIMESTEPS]
		idx += 1
		if idx < BATCH_SIZE:
			continue
		else:
			# Add 'lstm/lstm_0/control_dependency' if you want to dump previous layer
			# LSTM.
			lstm_emb = sess.run(t['lstm/lstm_1/control_dependency'],
								feed_dict={t['char_inputs_in']: char_ids_inputs,
											t['inputs_in']: inputs,
											t['targets_in']: targets,
											t['target_weights_in']: weights})
# 			lstm_emb = np.array(np.random.rand(idx, 128))
			idx = 0
			if len(final_vec) == 0:
				final_vec = lstm_emb
			else:
				final_vec = np.concatenate((final_vec, lstm_emb), axis=0)
	print('Before final size ', len(final_vec))
	if idx > 0:
		# Add 'lstm/lstm_0/control_dependency' if you want to dump previous layer
		# LSTM.
		lstm_emb = sess.run(t['lstm/lstm_1/control_dependency'],
							feed_dict={t['char_inputs_in']: char_ids_inputs,
										t['inputs_in']: inputs,
										t['targets_in']: targets,
										t['target_weights_in']: weights})
# 		lstm_emb = np.array(np.random.rand(idx, 128))
		if len(final_vec) == 0:
			final_vec = lstm_emb
		else:
			final_vec = np.concatenate((final_vec, lstm_emb), axis=0)
		print('Final size ', len(final_vec))
	return final_vec