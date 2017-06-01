'''
Created on Apr 23, 2017

@author: tonyq
'''
from tqdm._tqdm import tqdm
import pickle as pkl

with open('contencVocab.pkl', 'rb') as vocab_file:
	trainSet = pkl.load(vocab_file)

with open('testVocab.pkl', 'rb') as vocab_file:
	testSet = pkl.load(vocab_file)
	
contentSet = trainSet.union(testSet)

with open('../dsk16g/glove.840B.300d.txt', 'r', encoding='utf8') as fhd:
	with open('../dsk16g/glove.840B.quoraVocab.300d.txt', 'w', encoding='utf8') as fwrt:
		for line in tqdm(fhd):
			if line.strip().split(' ')[0] in contentSet:
				fwrt.write(line)
