'''
Created on Mar 17, 2017

@author: tonyq
'''

import pandas as pd
import numpy as np
import re, sys
from bs4 import BeautifulSoup
import logging
from keras.preprocessing.sequence import pad_sequences
from tqdm._tqdm import tqdm
from nltk.tokenize import word_tokenize
from numpy import array, zeros
import operator

logger = logging.getLogger(__name__)

uri_re = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'

def stripTagsAndUris(x):
	if x:
		# BeautifulSoup on content
		soup = BeautifulSoup(x, "html.parser")
		# Stripping all <code> tags with their content if any
		if soup.code:
			soup.code.decompose()
		# Get all the text out of the html
		text = soup.get_text()
		# Returning text stripping out all uris
		return re.sub(uri_re, "", text)
	else:
		return ""
	
def get_words(text):
# 	word_split = re.compile('[^a-zA-Z0-9_\\+\\-]')
# 	return [word.strip().lower() for word in word_split.split(text)]
	text = str(text)
	text = text.replace('’s', ' ’s')
	text = text.replace('…', ' ')
	text = text.replace('”', ' ')
	text = text.replace('“', ' ')
	text = text.replace('‘', ' ')
	text = text.replace('’', ' ')
	text = text.replace('"', ' ')
	text = text.replace("'", " ")
	text = text.replace('-', ' ')
	text = text.replace('/', ' ')
	text = text.replace("\\", " ")
	return word_tokenize(text)
	
def get_pdTable(path, notag=False):
	logger.info(' Processing pandas csv ')
	pdtable = pd.read_csv(path)
	if notag:
		return pdtable.id, pdtable.question1, pdtable.question2
	else:
		return pdtable.id, pdtable.question1, pdtable.question2, pdtable.is_duplicate

def tokenizeIt(table, clean=False):
	tokenizedTable = []
	maxLen = 0
	for text in tqdm(table, file=sys.stdout):
		if clean:
# 			text = stripTagsAndUris(text)
			text = get_words(text)
			tokenizedTable.append(text)
			if len(text) > maxLen:
				maxLen = len(text)
		else:
			text = str(text).split(' ')
			tokenizedTable.append(text)
			if len(text) > maxLen:
				maxLen = len(text)
	return tokenizedTable, maxLen		
	
def createVocab(tableList, min_count=1, reservedList=['<pad>', '<EOF>', '<unk>']):
	logger.info(' Creating vocabulary ')
	contentList = []
	for list1 in tableList:
		contentList.extend(list1)
	wdFrq = {}
	total_words = 0
	for line in contentList:
		for wd in line:
			try:
				wdFrq[wd] += 1
			except KeyError:
				wdFrq[wd] = 1
			total_words += 1
	logger.info('  %i total words, %i unique words ' % (total_words, len(wdFrq)))

	sorted_word_freqs = sorted(wdFrq.items(), key=operator.itemgetter(1), reverse=True)
	vocab_size = 0
	for _, freq in sorted_word_freqs:
		if freq >= min_count:
			vocab_size += 1
	vocabDict = {}
	vocabReverseDict = []
	idx = 0
	for item1 in reservedList:
		vocabDict[item1] = idx
		vocabReverseDict.append(item1)
		idx += 1
	for word, _ in sorted_word_freqs[:vocab_size]:
		vocabDict[word] = idx
		vocabReverseDict.append(word)
		idx += 1
		
	logger.info('  vocab size %i ' % len(vocabReverseDict))
	return vocabDict, vocabReverseDict

def word2num(contentTable, vocab, unk, maxLen, padding=None, eof=None):
	unk_hit = 0
	totalword = 0
	data = []
	for line in contentTable:
		w2num = []
		for word in line:			
			if word in vocab:
				w2num.append(vocab[word])
			else:
				if not type(unk) is type(None):
					w2num.append(vocab[unk])
				unk_hit += 1
			totalword += 1
		if not type(eof) is type(None):
			w2num.append(vocab[eof])
		data.append(w2num)
	logger.info('  total %i tokens processed, %i unk hit ' % (totalword, unk_hit))
	# pad to np array	
	if not type(padding) is type(None):
		logger.info('  padding data to width %d by %s padding' % (maxLen, padding))
		np_ary = pad_sequences(data, maxlen=maxLen, padding=padding)
	else:
		np_ary = array(data)
	return np_ary

def to_categorical2D(y, nb_classes=None):
	if not nb_classes:
		nb_classes = y.max()
	return (np.arange(nb_classes) == y[:,:,None]).astype(int)

def to_categoricalAll(y, nb_classes):
	categorical = zeros((len(y),nb_classes))
	line_idx = 0
	for line in y:
		for elem in line:
			categorical[line_idx][elem] = 1
		line_idx += 1
	return categorical

def categorical_toary(y, round01=False):
	(length, nb_classes) = y.shape
	if round01:
		y = np.around(y)
	y_ary = []
	for i in range(length):
		y_ary.append(np.argwhere(y[i,:] == 1).ravel().tolist())
	return y_ary

def prob_top_n(y, top=5):
	(length, nb_classes) = y.shape
	y_ary = []
	for i in range(length):
		idx_prob = list(zip(list(range(nb_classes)), y[i,:]))
		sorted_idx_prob = sorted(idx_prob, key=operator.itemgetter(1), reverse=True)[:top]
		idx_round = np.around(sorted_idx_prob).astype(int)
		idx_pos = []
		for (idx, prob) in idx_round:
			if prob == 1:
				idx_pos.append(idx)
		y_ary.append(idx_pos)
	return y_ary

def w2vEmbdReader(embd_path, reVocab, embd_dim):
	embd_matrix = np.zeros( (len(reVocab), embd_dim) )
	with open(embd_path, 'r', encoding='utf8') as fhd:
		idx = 1 # let 1st padding line all zeros
		for line in tqdm(fhd):
			elem = line.strip().split(' ')
			assert len(elem) == embd_dim + 1, 'Incorrect Embedding Dimension, expect %d but got %d ' % (embd_dim, len(elem)-1)
			w2vec = np.array(elem[1:])
			embd_matrix[idx] = w2vec
			idx += 1
	return embd_matrix