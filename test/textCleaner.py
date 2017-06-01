'''
Created on Apr 23, 2017

@author: tonyq
'''

import pandas as pd
from nltk.tokenize import word_tokenize
from tqdm._tqdm import tqdm
import csv

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

train = pd.read_csv("train.csv")
totalen = len(train.question1)
print('Total size: ', totalen)

fulllist = zip(train.id, train.question1, train.question2, train.is_duplicate)
length = len(train.id)
del train

with open("train.clean.csv", "w", encoding='utf8') as fwrt:
	writer_sub = csv.writer(fwrt)
	writer_sub.writerow(['id','question1','question2','is_duplicate'])
	for (theid, q1, q2, dup) in tqdm(fulllist, total=length):
		try:
			text_q1 = ' '.join(get_words(q1))
			text_q2 = ' '.join(get_words(q2))
		except AttributeError:
			print(theid)
			print(q1)
			print(q2)
			raise AttributeError
		writer_sub.writerow([theid, text_q1, text_q2, dup])
