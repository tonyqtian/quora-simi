'''
Created on Mar 17, 2017

@author: tonyq
'''
import pandas as pd
# import re
# from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from tqdm._tqdm import tqdm
import pickle as pkl

# uri_re = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'

# def stripTagsAndUris(x):
# 	if x:
# 		# BeautifulSoup on content
# 		soup = BeautifulSoup(x, "html.parser")
# 		# Stripping all <code> tags with their content if any
# 		if soup.code:
# 			soup.code.decompose()
# 		# Get all the text out of the html
# 		text = soup.get_text()
# 		# Returning text stripping out all uris
# 		return re.sub(uri_re, "", text)
# 	else:
# 		return ""
	
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
	text = text.replace('\\', ' ')
	return word_tokenize(text)
	
vocab = set()

# with open('../data/googleWord2Vec_300M_vocablist.txt', 'r', encoding='utf8') as fhd:
# 	for line in tqdm(fhd):
# 		vocab.add(line.strip())
# print(len(vocab))

# with open('../dsk16g/glove.twitter.27B.25d.txt', 'r', encoding='utf8') as fhd:
# 	for line in tqdm(fhd):
# 		vocab.add(line.strip().split(' ')[0])
# print(len(vocab))

train = pd.read_csv("train.csv")
totalen = len(train.question1)
print('Total size: ', totalen)

fulllist = zip(train.question1, train.question2)
length = len(train.question1)
del train
contentSet = set()
count = 0
for (q1, q2) in tqdm(fulllist, total=length):
	count += 1
	try:
		text = set(get_words(q1))
		contentSet.update(text)
	except AttributeError:
		print(count)
		print(q1)
		print(q2)
		raise AttributeError
	try:
		text = set(get_words(q2))
		contentSet.update(text)
	except AttributeError:
		print(count)
		print(q1)
		print(q2)
		raise AttributeError

with open('../output/contencVocab.pkl', 'wb') as vocab_file:
	pkl.dump(contentSet, vocab_file)

with open('contencVocab.pkl', 'rb') as vocab_file:
	trainSet = pkl.load(vocab_file)

with open('testVocab.pkl', 'rb') as vocab_file:
	testSet = pkl.load(vocab_file)
	
contentSet = trainSet.union(testSet)
# contentSet =testSet
contentHit = contentSet.intersection(vocab)
	
print('Hit content %d' % len(contentHit))
# print(contentHit)
print('Hit rate %.4f  Vocab size %d' % (len(contentHit)/len(contentSet), len(contentSet)))
print(contentSet.difference(vocab))