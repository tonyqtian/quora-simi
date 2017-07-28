'''
Created on Apr 18, 2017

@author: tonyq
'''
import matplotlib
matplotlib.use('Agg')

from tqdm._tqdm import tqdm
import logging, time
import pickle as pkl
import csv
from numpy import array, squeeze, vstack, inf, nan, concatenate, random, vstack, ones
from pandas import read_csv, DataFrame

from util.utils import setLogger, mkdir, print_args
from util.data_processing import get_pdTable, text_cleaner, embdReader, tokenizeIt, csv_processing
from util.my_layers import DenseWithMasking, Conv1DWithMasking, MaxOverTime, MeanOverTime
from util.model_eval import PlotPic
from src.lm_1b_model import lm_1b_infer
	
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.utils import plot_model
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing.data import StandardScaler
from keras.callbacks import ModelCheckpoint, EarlyStopping

# choose model 
from src.rnn_model import getModel

logger = logging.getLogger(__name__)
MAX_NB_WORDS = 200000

def train(args):
	timestr = time.strftime("%Y%m%d-%H%M%S-")
	output_dir = args.out_dir_path + '/' + time.strftime("%m%d")
	mkdir(output_dir)
	setLogger(timestr, out_dir=output_dir)
	print_args(args)
	
	if args.load_input_pkl is '':
		# process train and test data
		logger.info('Loading training file...')
		_, train_question1, train_question2, train_y = get_pdTable(args.train_path)
		# train_question1, train_question2, train_y = csv_processing(args.train_path)
		logger.info('Train csv: %d line loaded ' % len(train_question1))
		logger.info('Loading test file...')
		test_ids, test_question1, test_question2 = get_pdTable(args.test_path, notag=True)
		# test_question1, test_question2, test_ids = csv_processing(args.test_path, test=True)
		logger.info('Test csv: %d line loaded ' % len(test_question1))
	
		logger.info('Text cleaning... ')
		train_question1, train_maxLen1 = text_cleaner(train_question1)
		train_question2, train_maxLen2 = text_cleaner(train_question2)
		test_question1, test_maxLen1 = text_cleaner(test_question1)
		test_question2, test_maxLen2 = text_cleaner(test_question2)
	# 	train_question1, train_maxLen1 = tokenizeIt(train_question1, clean=args.rawMaterial)
	# 	train_question2, train_maxLen2 = tokenizeIt(train_question2, clean=args.rawMaterial)
	# 	test_question1, test_maxLen1 = tokenizeIt(test_question1, clean=args.rawMaterial)
	# 	test_question2, test_maxLen2 = tokenizeIt(test_question2, clean=args.rawMaterial)
		inputLength = max(train_maxLen1, train_maxLen2, test_maxLen1, test_maxLen2)
		logger.info('Max input length: %d ' % inputLength)
		inputLength = 30
		logger.info('Reset max length to 30')
	
		tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
		tokenizer.fit_on_texts(train_question1 + train_question2 + test_question1 + test_question2)
		
		sequences_1 = tokenizer.texts_to_sequences(train_question1)
		sequences_2 = tokenizer.texts_to_sequences(train_question2)
		test_sequences_1 = tokenizer.texts_to_sequences(test_question1)
		test_sequences_2 = tokenizer.texts_to_sequences(test_question2)
		del train_question1, train_question2, test_question1, test_question2
		
		word_index = tokenizer.word_index
		logger.info('Found %s unique tokens' % len(word_index))
		
		train_x1 = pad_sequences(sequences_1, maxlen=inputLength)
		train_x2 = pad_sequences(sequences_2, maxlen=inputLength)
		train_y = array(train_y)
		logger.info('Shape of data tensor: (%d, %d)' % train_x1.shape)
		logger.info('Shape of label tensor: (%d, )' % train_y.shape)
		
		test_x1 = pad_sequences(test_sequences_1, maxlen=inputLength)
		test_x2 = pad_sequences(test_sequences_2, maxlen=inputLength)
		test_ids = array(test_ids)
		del sequences_1, sequences_2, test_sequences_1, test_sequences_2
		with open(output_dir + '/'+ timestr + 'input_train_test.pkl', 'wb') as input_file:
			logger.info('Dumping processed input to pickle...')
			pkl.dump((train_x1, train_x2, train_y, test_x1, test_x2, test_ids, word_index), input_file)
	else:
		with open(args.load_input_pkl, 'rb') as input_file:
			train_x1, train_x2, train_y, test_x1, test_x2, test_ids, word_index = pkl.load(input_file)
			logger.info('Shape of data tensor: (%d, %d)' % train_x1.shape)
			logger.info('Shape of label tensor: (%d, )' % train_y.shape)
		inputLength = 30
		logger.info('Reset max length to 30')
			
	if args.w2v:
		if args.w2v.endswith('.pkl'):
			with open(args.w2v, 'rb') as embd_file:
				logger.info('Loading word embedding from pickle...')
				embdw2v, vocabReverseDict = pkl.load(embd_file)
				if not len(vocabReverseDict) == len(word_index):
					logger.info('WARNING: reversed dict len incorrect %d , but word dict len %d ' % \
							(len(vocabReverseDict), len(word_index)))
		else: 
			logger.info('Loading word embedding from text file...')
			embdw2v, vocabReverseDict = embdReader(args.w2v, args.embd_dim, word_index, MAX_NB_WORDS)
			with open(output_dir + '/'+ timestr + 'embd_dump.' + str(args.embd_dim) + 'd.pkl', 'wb') as embd_file:
				logger.info('Dumping word embedding to pickle...')
				pkl.dump((embdw2v, vocabReverseDict), embd_file)

# 	if args.load_vocab_from_file:
# 		with open(args.load_vocab_from_file, 'rb') as vocab_file:
# 			(vocabDict, vocabReverseDict) = pkl.load(vocab_file)
# 			unk = None
# 			if args.w2v:
# 				if args.w2v.endswith('.pkl'):
# 					with open(args.w2v, 'rb') as embd_file:
# 						embdw2v = pkl.load(embd_file)
# 				else:
# 					from util.data_processing import w2vEmbdReader
# 					embdw2v = w2vEmbdReader(args.w2v, vocabReverseDict, args.embd_dim)
# 					with open(output_dir + '/'+ timestr + 'embd_dump.' + str(args.embd_dim) + 'd.pkl', 'wb') as embd_file:
# 						pkl.dump(embdw2v, embd_file)
# 			else:
# 				embdw2v = None
# 	else:
# 		from util.data_processing import createVocab
# 		vocabDict, vocabReverseDict = createVocab([train_question1, train_question2, test_question1, test_question2], 
# 												min_count=3, reservedList=['<pad>', '<unk>'])
# 		embdw2v = None
# 		unk = '<unk>'
## 	logger.info(vocabDict)
	
# 	# word to padded numerical np array
# 	from util.data_processing import word2num
# 	train_x1 = word2num(train_question1, vocabDict, unk, inputLength, padding='pre')
# 	train_x2 = word2num(train_question2, vocabDict, unk, inputLength, padding='pre')
# 	test_x1 = word2num(test_question1, vocabDict, unk, inputLength, padding='pre')
# 	test_x2 = word2num(test_question2, vocabDict, unk, inputLength, padding='pre')

	# Loading train features
	if args.train_feature_path is not '':
		logger.info('Loading train features from file %s ' % args.train_feature_path)
		df_train = read_csv(args.train_feature_path, encoding="ISO-8859-1")
		if args.feature_list is not '':
			feature_list = args.feature_list.split(',')
			train_features = DataFrame()
			for feature_name in feature_list:
				train_features[feature_name.strip()] = df_train[feature_name.strip()]
		elif args.fidx_end == 0:
			train_features = df_train.iloc[:, args.fidx_start:]
		else:
			train_features = df_train.iloc[:, args.fidx_start:args.fidx_end]

		if args.train_bowl_feature_path is not '':
			logger.info('Loading train 1bowl features from file %s ' % args.train_bowl_feature_path)
			df_train = read_csv(args.train_bowl_feature_path, encoding="ISO-8859-1")
			if args.bowl_feat_list is not '':
				bowl_feat_list = args.bowl_feat_list.split(',')
				for feature_name in bowl_feat_list:
					train_features[feature_name.strip()] = df_train[feature_name.strip()]
			else:
				for feature_name in df_train.columns:
					if feature_name.startswith('z_'):
						train_features[feature_name] = df_train[feature_name]

		logger.info('Final train feature list: \n %s ' % ','.join(list(train_features.columns.values)))
		feature_length = len(train_features.columns)
		train_features = train_features.replace([inf, -inf, nan], 0)
		train_features = array(train_features)
		logger.info('Loaded train feature shape: (%d, %d) ' % train_features.shape)
		del df_train

		# Loading test features
		logger.info('Loading test features from file %s ' % args.test_feature_path)
		df_test = read_csv(args.test_feature_path, encoding="ISO-8859-1")
		if args.feature_list is not '':
			feature_list = args.feature_list.split(',')
			test_features = DataFrame()
			for feature_name in feature_list:
				test_features[feature_name.strip()] = df_test[feature_name.strip()]
		elif args.fidx_end == 0:
			test_features = df_test.iloc[:, args.fidx_start:]
		else:
			test_features = df_test.iloc[:, args.fidx_start:args.fidx_end]

		if args.test_bowl_feature_path is not '':
			logger.info('Loading test 1bowl features from file %s ' % args.test_bowl_feature_path)
			df_test = read_csv(args.test_bowl_feature_path, encoding="ISO-8859-1")
			if args.bowl_feat_list is not '':
				bowl_feat_list = args.bowl_feat_list.split(',')
				for feature_name in bowl_feat_list:
					test_features[feature_name.strip()] = df_test[feature_name.strip()]
			else:
				for feature_name in df_test.columns:
					if feature_name.startswith('z_'):
						test_features[feature_name] = df_test[feature_name]

		test_features = test_features.replace([inf, -inf, nan], 0)
		test_features = array(test_features)
		logger.info('Loaded test feature shape: (%d, %d) ' % test_features.shape)
		del df_test

		# Normalize Data
		ss = StandardScaler()
		ss.fit(vstack((train_features, test_features)))
		train_features = ss.transform(train_features)
		test_features = ss.transform(test_features)
		del ss
		logger.info('Features normalized ')

	train_x1_aug = vstack((train_x1, train_x2))
	train_x2_aug = vstack((train_x2, train_x1))
	train_y = concatenate((train_y, train_y))
	train_x = [train_x1_aug, train_x2_aug]
	test_x = [test_x1, test_x2]
	if args.train_feature_path is not '':
		train_features = vstack((train_features, train_features))
		train_x += [train_features]
		test_x +=[test_features]

	# ########################################
	# ## sample train/validation data
	# ########################################
	# # np.random.seed(1234)
	# perm = random.permutation(len(train_x1))
	# idx_train = perm[:int(len(train_x1) * (1 - args.valid_split))]
	# idx_val = perm[int(len(train_x1) * (1 - args.valid_split)):]
	#
	# data_1_train = vstack((train_x1[idx_train], train_x2[idx_train]))
	# data_2_train = vstack((train_x2[idx_train], train_x1[idx_train]))
	# leaks_train = vstack((train_features[idx_train], train_features[idx_train]))
	# labels_train = concatenate((train_y[idx_train], train_y[idx_train]))
	#
	# data_1_val = vstack((train_x1[idx_val], train_x2[idx_val]))
	# data_2_val = vstack((train_x2[idx_val], train_x1[idx_val]))
	# leaks_val = vstack((train_features[idx_val], train_features[idx_val]))
	# labels_val = concatenate((train_y[idx_val], train_y[idx_val]))

	# re_weight = True  # whether to re-weight classes to fit the 17.5% share in test set
	# weight_val = ones(len(labels_val))
	# if re_weight:
	# 	weight_val *= 0.472001959
	# 	weight_val[labels_val == 0] = 1.309028344

	########################################
	## add class weight
	########################################
	if args.re_weight:
		class_weight = {0: 1.309028344, 1: 0.472001959}
	else:
		class_weight = None

	# 	# Dump vocab
# 	if not args.load_vocab_from_file:
# 		with open(output_dir + '/'+ timestr + 'vocab.pkl', 'wb') as vocab_file:
# 			pkl.dump((vocabDict, vocabReverseDict), vocab_file)

	if args.load_model_json:
		with open(args.load_model_json, 'r') as json_file:
			rnnmodel = model_from_json(json_file.read(), 
									custom_objects={"DenseWithMasking": DenseWithMasking,
													"Conv1DWithMasking": Conv1DWithMasking,
													"MaxOverTime": MaxOverTime, 
													"MeanOverTime": MeanOverTime})
		logger.info('Loaded model from saved json')
	else:
		if args.train_feature_path is not '':
			rnnmodel = getModel(args, inputLength, len(word_index)+1, embd=embdw2v, feature_length=feature_length)
		else:
			rnnmodel = getModel(args, inputLength, len(word_index)+1, embd=embdw2v)
		
	if args.load_model_weights:
		rnnmodel.load_weights(args.load_model_weights)
		logger.info('Loaded model from saved weights')
		
	if args.optimizer == 'rmsprop':
		optimizer = RMSprop(lr=args.learning_rate)
	else:
		optimizer = args.optimizer

	myMetrics = 'acc' # 'binary_accuracy' # 'mse'
	rnnmodel.compile(loss=args.loss, optimizer=optimizer, metrics=[myMetrics])
	rnnmodel.summary()

	if args.save_model:
		## Plotting model
		logger.info('Plotting model architecture')
		plot_model(rnnmodel, to_file = output_dir + '/' + timestr + 'model_plot.png')
		logger.info('  Done')
		
		## Save model architecture
		logger.info('Saving model architecture')
		with open(output_dir + '/'+ timestr + 'model_config.json', 'w') as arch:
			arch.write(rnnmodel.to_json(indent=2))
		logger.info('  Done')
			
	# train and test model
	myCallbacks = []
# 	if not args.predict_test:
# 		if args.eval_on_epoch:
# 			from util.model_eval import Evaluator
# 			evl = Evaluator(args, output_dir, timestr, myMetrics, test_x, test_y, vocabReverseDict)
# 			myCallbacks.append(evl)
	if args.save_model:
		bst_model_path = output_dir + '/' + timestr + 'best_model_weights.h5'
		model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True, verbose=1)
		myCallbacks.append(model_checkpoint)
	if args.plot:
		plot_pic = PlotPic(args, output_dir, timestr, myMetrics)
		myCallbacks.append(plot_pic)
	if args.earlystop:
		earlystop = EarlyStopping(patience = args.earlystop, verbose=1, mode='auto')
		myCallbacks.append(earlystop)
			
	rnnmodel.fit(train_x, train_y, validation_split=args.valid_split, batch_size=args.train_batch_size,
				 epochs=args.epochs, class_weight=class_weight, callbacks=myCallbacks)
	# rnnmodel.fit([data_1_train, data_2_train, leaks_train], labels_train,
	# 				 validation_data=([data_1_val, data_2_val, leaks_val], labels_val, weight_val),
	# 				 epochs=args.epochs, batch_size=args.train_batch_size, shuffle=True,
	# 				 class_weight=class_weight, callbacks=myCallbacks)

	if args.predict_test:
		logger.info("Tuning model to best record...")
		rnnmodel.load_weights(bst_model_path)
		logger.info("Predicting test file result...")
		preds = rnnmodel.predict(test_x, batch_size=args.eval_batch_size, verbose=1)
		preds = squeeze(preds)
		logger.info('Write predictions into file... Total line: %d ' % len(preds))
		with open(output_dir + '/'+ timestr + 'predict.csv', 'w', encoding='utf8') as fwrt:
			writer_sub = csv.writer(fwrt)
			writer_sub.writerow(['test_id', 'is_duplicate'])
			idx = 0
			for itm in tqdm(preds):
				writer_sub.writerow([idx, itm])
				idx += 1
# 	elif not args.eval_on_epoch:
# 		rnnmodel.evaluate(test_x, test_y, batch_size=args.eval_batch_size)
	
	# test output (remove duplicate, remove <pad> <unk>, comparable layout, into csv)
	# final inference: output(remove duplicate, remove <pad> <unk>, limit output words to 3 or 2 or 1..., into csv)
	
	
def inference(args):

	timestr = time.strftime("%Y%m%d-%H%M%S-")
	output_dir = args.out_dir_path + '/' + time.strftime("%m%d")
	mkdir(output_dir)
	setLogger(timestr, out_dir=output_dir)
	print_args(args)

	if args.load_input_pkl is '':
		raise NotImplementedError('only support loading testing materials from pickle')
	else:
		with open(args.load_input_pkl, 'rb') as input_file:
			train_x1, train_x2, train_y, test_x1, test_x2, test_ids, word_index = pkl.load(input_file)
			logger.info('Shape of test data tensor: (%d, %d)' % test_x1.shape)

	# Loading train features
	if args.train_feature_path is not '':
		logger.info('Loading train features from file %s ' % args.train_feature_path)
		df_train = read_csv(args.train_feature_path, encoding="ISO-8859-1")
		if args.feature_list is not '':
			feature_list = args.feature_list.split(',')
			train_features = DataFrame()
			for feature_name in feature_list:
				train_features[feature_name.strip()] = df_train[feature_name.strip()]
		elif args.fidx_end == 0:
			train_features = df_train.iloc[:, args.fidx_start:]
		else:
			train_features = df_train.iloc[:, args.fidx_start:args.fidx_end]

		if args.train_bowl_feature_path is not '':
			logger.info('Loading train 1bowl features from file %s ' % args.train_bowl_feature_path)
			df_train = read_csv(args.train_bowl_feature_path, encoding="ISO-8859-1")
			if args.bowl_feat_list is not '':
				bowl_feat_list = args.bowl_feat_list.split(',')
				for feature_name in bowl_feat_list:
					train_features[feature_name.strip()] = df_train[feature_name.strip()]
			else:
				for feature_name in df_train.columns:
					if feature_name.startswith('z_'):
						train_features[feature_name] = df_train[feature_name]

		logger.info('Final train feature list: \n %s ' % ','.join(list(train_features.columns.values)))
		feature_length = len(train_features.columns)
		train_features = train_features.replace([inf, -inf, nan], 0)
		train_features = array(train_features)
		logger.info('Loaded train feature shape: (%d, %d) ' % train_features.shape)
		del df_train

		# Loading test features
		logger.info('Loading test features from file %s ' % args.test_feature_path)
		df_test = read_csv(args.test_feature_path, encoding="ISO-8859-1")
		if args.feature_list is not '':
			feature_list = args.feature_list.split(',')
			test_features = DataFrame()
			for feature_name in feature_list:
				test_features[feature_name.strip()] = df_test[feature_name.strip()]
		elif args.fidx_end == 0:
			test_features = df_test.iloc[:, args.fidx_start:]
		else:
			test_features = df_test.iloc[:, args.fidx_start:args.fidx_end]

		if args.test_bowl_feature_path is not '':
			logger.info('Loading test 1bowl features from file %s ' % args.test_bowl_feature_path)
			df_test = read_csv(args.test_bowl_feature_path, encoding="ISO-8859-1")
			if args.bowl_feat_list is not '':
				bowl_feat_list = args.bowl_feat_list.split(',')
				for feature_name in bowl_feat_list:
					test_features[feature_name.strip()] = df_test[feature_name.strip()]
			else:
				for feature_name in df_test.columns:
					if feature_name.startswith('z_'):
						test_features[feature_name] = df_test[feature_name]

		test_features = test_features.replace([inf, -inf, nan], 0)
		test_features = array(test_features)
		logger.info('Loaded test feature shape: (%d, %d) ' % test_features.shape)
		del df_test

		# Normalize Data
		ss = StandardScaler()
		ss.fit(vstack((train_features, test_features)))
		# train_features = ss.transform(train_features)
		test_features = ss.transform(test_features)
		del ss
		logger.info('Test Features normalized ')

	test_x = [test_x1, test_x2]
	if args.test_feature_path is not '':
		test_x +=[test_features]

	if args.load_model_json:
		with open(args.load_model_json, 'r') as json_file:
			rnnmodel = model_from_json(json_file.read(),
									custom_objects={"DenseWithMasking": DenseWithMasking,
													"Conv1DWithMasking": Conv1DWithMasking,
													"MaxOverTime": MaxOverTime,
													"MeanOverTime": MeanOverTime})
		logger.info('Loaded model from saved json')

	if args.load_model_weights:
		logger.info('Loading model from saved weights')
		rnnmodel.load_weights(args.load_model_weights)

	if args.predict_test:
		logger.info("Predicting test file result...")
		preds = rnnmodel.predict(test_x, batch_size=args.eval_batch_size, verbose=1)
		preds = squeeze(preds)
		logger.info('Write predictions into file... Total line: %d ' % len(preds))
		with open(output_dir + '/'+ timestr + 'predict.csv', 'w', encoding='utf8') as fwrt:
			writer_sub = csv.writer(fwrt)
			writer_sub.writerow(['test_id', 'is_duplicate'])
			idx = 0
			for itm in tqdm(preds):
				writer_sub.writerow([idx, itm])
				idx += 1

	# raise NotImplementedError
	# timestr = time.strftime("%Y%m%d-%H%M%S-")
	# output_dir = args.out_dir_path + '/' + time.strftime("%m%d")
	# mkdir(output_dir)
	# setLogger(timestr, out_dir=output_dir)
	# print_args(args)
	#
	# # process train and test data
	# _, train_question1, train_question2, train_y = get_pdTable(args.train_path)
	# _, test_question1, test_question2, test_y = get_pdTable(args.test_path)
	#
	# train_question1, train_maxLen1 = tokenizeIt(train_question1, clean=args.rawMaterial, addHead='<s>')
	# train_question2, train_maxLen2 = tokenizeIt(train_question2, clean=args.rawMaterial, addHead='<s>')
	# test_question1, test_maxLen1 = tokenizeIt(test_question1, clean=args.rawMaterial, addHead='<s>')
	# test_question2, test_maxLen2 = tokenizeIt(test_question2, clean=args.rawMaterial, addHead='<s>')
	# inputLength = max(train_maxLen1, train_maxLen2, test_maxLen1, test_maxLen2)
	# print('Max input length: ', inputLength)
	# inputLength = 50
	# print('Reset max length to %d' % inputLength)
	#
	# train_question1_vec = lm_1b_infer(args, inputLength, train_question1)
	# print('Train Q1 shape: ', train_question1_vec.shape)
	# train_question2_vec = lm_1b_infer(args, inputLength, train_question2)
	# print('Train Q2 shape: ', train_question2_vec.shape)
	#
	# with open(output_dir + '/' + timestr + 'train_lstm_vec.pkl', 'wb') as f:
	# 	pkl.dump( (train_question1_vec, train_question2_vec, train_y), f)
	# 	print('Training LSTM embedding file saved.')
	#
	# test_question1_vec = lm_1b_infer(args, inputLength, test_question1)
	# print('Test Q1 shape: ', test_question1_vec.shape)
	# test_question2_vec = lm_1b_infer(args, inputLength, test_question2)
	# print('Test Q2 shape: ', test_question2_vec.shape)
	#
	# with open(output_dir + '/' + timestr + 'test_lstm_vec.pkl', 'wb') as f:
	# 	pkl.dump( (test_question1_vec, test_question2_vec, test_y), f)
	# 	print('Training LSTM embedding file saved.')
	