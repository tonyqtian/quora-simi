'''
Created on Apr 18, 2017

@author: tonyq
'''
import matplotlib
from tqdm._tqdm import tqdm
matplotlib.use('Agg')

import logging, time
import pickle as pkl
from util.utils import setLogger, mkdir, print_args
from util.data_processing import get_pdTable, tokenizeIt, createVocab, word2num, w2vEmbdReader

logger = logging.getLogger(__name__)

def train(args):
	timestr = time.strftime("%Y%m%d-%H%M%S-")
	output_dir = args.out_dir_path + '/' + time.strftime("%m%d")
	mkdir(output_dir)
	setLogger(timestr, out_dir=output_dir)
	print_args(args)
	
	# process train and test data
	_, train_question1, train_question2, train_y = get_pdTable(args.train_path)
	if args.predict_test:
		_, test_question1, test_question2 = get_pdTable(args.test_path, notag=True)
	else:
		_, test_question1, test_question2, test_y = get_pdTable(args.test_path)
	
	train_question1, train_maxLen1 = tokenizeIt(train_question1, clean=args.rawMaterial)
	train_question2, train_maxLen2 = tokenizeIt(train_question2, clean=args.rawMaterial)
	test_question1, test_maxLen1 = tokenizeIt(test_question1, clean=args.rawMaterial)
	test_question2, test_maxLen2 = tokenizeIt(test_question2, clean=args.rawMaterial)
	inputLength = max(train_maxLen1, train_maxLen2, test_maxLen1, test_maxLen2)
	print('Max input length: ', inputLength)
	inputLength = 32
	print('Reset max length to 32')
		
	if args.load_vocab_from_file:
		with open(args.load_vocab_from_file, 'rb') as vocab_file:
			(vocabDict, vocabReverseDict) = pkl.load(vocab_file)
			unk = None
			if args.w2v:
				embdw2v = w2vEmbdReader(args.w2v, vocabReverseDict, args.embd_dim)
			else:
				embdw2v = None
	else:
		vocabDict, vocabReverseDict = createVocab([train_question1, train_question2, test_question1, test_question2], 
												min_count=3, reservedList=['<pad>', '<unk>'])
		embdw2v = None
		unk = '<unk>'
	# logger.info(vocabDict)
	
	# word to padded numerical np array
	train_x1 = word2num(train_question1, vocabDict, unk, inputLength, padding='pre')
	train_x2 = word2num(train_question2, vocabDict, unk, inputLength, padding='pre')
	test_x1 = word2num(test_question1, vocabDict, unk, inputLength, padding='pre')
	test_x2 = word2num(test_question2, vocabDict, unk, inputLength, padding='pre')
	
	# choose model 
	from src.rnn_model import getModel
	
	# Dump vocab
	if not args.load_vocab_from_file:
		with open(output_dir + '/'+ timestr + 'vocab.pkl', 'wb') as vocab_file:
			pkl.dump((vocabDict, vocabReverseDict), vocab_file)
			
	if args.load_model_json:
		from keras.models import model_from_json
		from util.my_layers import DenseWithMasking
		with open(args.load_model_json, 'r') as json_file:
			rnnmodel = model_from_json(json_file.read(), custom_objects={"DenseWithMasking": DenseWithMasking})
		logger.info('Loaded model from saved json')
	else:
		rnnmodel = getModel(args, inputLength, len(vocabDict), embd=embdw2v)
		
	if args.load_model_weights:
		rnnmodel.load_weights(args.load_model_weights)
		logger.info('Loaded model from saved weights')
		
	if args.optimizer == 'rmsprop':
		from keras.optimizers import RMSprop
		optimizer = RMSprop(lr=args.learning_rate)
	else:
		optimizer = args.optimizer

	myMetrics = 'mse' #'binary_accuracy'
	rnnmodel.compile(loss=args.loss, optimizer=optimizer, metrics=[myMetrics])
	rnnmodel.summary()

	if args.save_model:
		## Plotting model
		logger.info('Plotting model architecture')
		from keras.utils import plot_model
		plot_model(rnnmodel, to_file = output_dir + '/' + timestr + 'model_plot.png')
		logger.info('  Done')
		
		## Save model architecture
		logger.info('Saving model architecture')
		with open(output_dir + '/'+ timestr + 'model_config.json', 'w') as arch:
			arch.write(rnnmodel.to_json(indent=2))
		logger.info('  Done')
	
	# train and test model
	myCallbacks = []
	if not args.predict_test:
		if args.eval_on_epoch:
			from util.model_eval import Evaluator
			evl = Evaluator(args, output_dir, timestr, myMetrics, [test_x1, test_x2], test_y, vocabReverseDict)
			myCallbacks.append(evl)
	if args.earlystop:
		from keras.callbacks import EarlyStopping
		earlystop = EarlyStopping(patience = args.earlystop, verbose=1, mode='auto')
		myCallbacks.append(earlystop)
	rnnmodel.fit([train_x1, train_x2], train_y, validation_split=args.valid_split, batch_size=args.train_batch_size, epochs=args.epochs, callbacks=myCallbacks)
	if args.predict_test:
		print("Predicting test file result...")
		preds = rnnmodel.predict([test_x1, test_x2], batch_size=args.eval_batch_size, verbose=1)
		from numpy import squeeze
		preds = squeeze(preds)
		print('Write predictions into file... Total line: ', len(preds))
		import csv
		with open(output_dir + '/'+ timestr + 'predict.csv', 'w', encoding='utf8') as fwrt:
			writer_sub = csv.writer(fwrt)
			writer_sub.writerow(['test_id', 'is_duplicate'])
			idx = 0
			for itm in tqdm(preds):
				writer_sub.writerow([idx, itm])
				idx += 1
	elif not args.eval_on_epoch:
		rnnmodel.evaluate([test_x1, test_x2], test_y, batch_size=args.eval_batch_size)
	
	# test output (remove duplicate, remove <pad> <unk>, comparable layout, into csv)
	# final inference: output(remove duplicate, remove <pad> <unk>, limit output words to 3 or 2 or 1..., into csv)
	
	
def inference(args):
	
	timestr = time.strftime("%Y%m%d-%H%M%S-")
	output_dir = args.out_dir_path + '/' + time.strftime("%m%d")
	mkdir(output_dir)
	setLogger(timestr, out_dir=output_dir)
	print_args(args)
	
	# process train and test data
	_, train_question1, train_question2, train_y = get_pdTable(args.train_path)
	_, test_question1, test_question2, test_y = get_pdTable(args.test_path)
	
	train_question1, train_maxLen1 = tokenizeIt(train_question1, clean=args.rawMaterial, addHead='<s>')
	train_question2, train_maxLen2 = tokenizeIt(train_question2, clean=args.rawMaterial, addHead='<s>')
	test_question1, test_maxLen1 = tokenizeIt(test_question1, clean=args.rawMaterial, addHead='<s>')
	test_question2, test_maxLen2 = tokenizeIt(test_question2, clean=args.rawMaterial, addHead='<s>')
	inputLength = max(train_maxLen1, train_maxLen2, test_maxLen1, test_maxLen2)
	print('Max input length: ', inputLength)
	inputLength = 50
	print('Reset max length to %d' % inputLength)

	from src.lm_1b_model import lm_1b_infer
	train_question1_vec = lm_1b_infer(args, inputLength, train_question1)
	print('Train Q1 shape: ', train_question1_vec.shape)
	train_question2_vec = lm_1b_infer(args, inputLength, train_question2)
	print('Train Q2 shape: ', train_question2_vec.shape)
	
	with open(output_dir + '/' + timestr + 'train_lstm_vec.pkl', 'wb') as f:
		pkl.dump( (train_question1_vec, train_question2_vec, train_y), f)
		print('Training LSTM embedding file saved.')
	
	test_question1_vec = lm_1b_infer(args, inputLength, test_question1)
	print('Test Q1 shape: ', test_question1_vec.shape)
	test_question2_vec = lm_1b_infer(args, inputLength, test_question2)
	print('Test Q2 shape: ', test_question2_vec.shape)
	
	with open(output_dir + '/' + timestr + 'test_lstm_vec.pkl', 'wb') as f:
		pkl.dump( (test_question1_vec, test_question2_vec, test_y), f)
		print('Training LSTM embedding file saved.')
	