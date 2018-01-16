'''
Created on Apr 23, 2017

@author: tonyq
'''

import pandas as pd
# from nltk.tokenize import word_tokenize
# from tqdm._tqdm import tqdm
# import csv
import re
import operator
from keras.preprocessing.text import Tokenizer

MAX_NB_WORDS = 200000


def get_words(text):
# 	word_split = re.compile('[^a-zA-Z0-9_\\+\\-]')
# 	return [word.strip().lower() for word in word_split.split(text)]
    text = str(text)
# 	text = text.replace('’s', ' ’s')
#     text = text.replace('…', ' ')
#     text = text.replace('”', ' ')
#     text = text.replace('“', ' ')
#     text = text.replace('‘', ' ')
#     text = text.replace('’', ' ')
#     text = text.replace('"', ' ')
# 	text = text.replace("'", " ")
#     text = text.replace('-', ' ')
# 	text = text.replace('/', ' ')
#     text = text.replace("\\", " ")
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!'.+-=%]", " ", text)
    text = text.replace("what's", "what is ")
    text = text.replace("'s", " ")
    text = text.replace("'ve", " have ")
    text = text.replace("can't", "cannot ")
    text = text.replace("n't", " not ")
    text = text.replace("i'm", "i am ")
    text = text.replace("'re", " are ")
    text = text.replace("'d", " would ")
    text = text.replace("'ll", " will ")
    text = text.replace(",", " , ")
    text = text.replace(".", " . ")
    text = text.replace("!", " ! ")
    # text = text.replace("/", " ")
    # text = text.replace("^", " ^ ")
    # text = text.replace("+", " + ")
    # text = text.replace("-", " - ")
    # text = text.replace("=", " = ")
    text = text.replace("'", " ")
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = text.replace(":", " : ")
    text = text.replace(" e g ", " eg ")
    text = text.replace(" b g ", " bg ")
    text = text.replace(" u s ", " american ")
    # text = re.sub(r"\0s", "0", text)
    text = text.replace(" 9 11 ", "911")
    text = text.replace("e - mail", "email")
    # text = text.replace("j k", "jk")
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()


test = pd.read_csv("../../quora-data/test.csv")
totalen = len(test.question1)
print('Test size: ', totalen)

train = pd.read_csv("../../quora-data/train.csv")
totalen = len(train.question1)
print('Train size: ', totalen)

text_material = test.question1 + test.question2 + train.question1 + train.question2
# text_material = train.question2[:1000]

train_material = []
for line in text_material:
    line = get_words(line)
    train_material.append(line)

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True)
tokenizer.fit_on_texts(train_material)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

sorted_word_freqs = sorted(tokenizer.word_counts.items(), key=operator.itemgetter(1), reverse=True)
print(sorted_word_freqs[-500:])

word_set = set([])
for ky, vl in tokenizer.word_counts.items():
    word_set.add(ky)
print('Total word in count %d' % len(word_set))

fasttext_set = set([])
with open('/data2/tonyq/quora-data/wiki-news-300d-1M-subword.vec') as f:
    next(f)
    for line in f:
        fasttext_set.add(line.split(' ')[0])
print("Total word in fasttext count %d" % len(fasttext_set))

covered = len(word_set.intersection(fasttext_set))
uncovered = len(word_set.difference(fasttext_set))
print("Quora word in fasttext %d (%.2f)" % (covered, 100*covered/len(word_set)))
print("Quora word not in fasttext %d (%.2f)" % (uncovered, 100*uncovered/len(word_set)))

w2v_set = set([])
with open('/data2/tonyq/quora-data/glove.840B.quoraVocab.300d.txt') as f:
    for line in f:
        w2v_set.add(line.split(' ')[0])
print("Total word in word2vec count %d" % len(w2v_set))

covered = len(word_set.intersection(w2v_set))
uncovered = len(word_set.difference(w2v_set))
print("Quora word in word2vec %d (%.2f)" % (covered, 100*covered/len(word_set)))
print("Quora word not in word2vec %d (%.2f)" % (uncovered, 100*uncovered/len(word_set)))

# # fulllist = zip(train.id, train.question1, train.question2, train.is_duplicate)
# fulllist = zip(train.test_id, train.question1, train.question2)
# length = len(train.question1)
# del train

# with open("test.clean.csv", "w", encoding='utf8') as fwrt:
#     writer_sub = csv.writer(fwrt)
# # 	writer_sub.writerow(['id','question1','question2','is_duplicate'])
#     writer_sub.writerow(['test_id','question1','question2'])
# # 	for (theid, q1, q2, dup) in tqdm(fulllist, total=length):
#     for (theid, q1, q2) in tqdm(fulllist, total=length):
#         try:
#             text_q1 = ' '.join(get_words(q1))
#             text_q2 = ' '.join(get_words(q2))
#         except AttributeError:
#             print(theid)
#             print(q1)
#             print(q2)
#             raise AttributeError
# # 		writer_sub.writerow([theid, text_q1, text_q2, dup])
#         writer_sub.writerow([theid, text_q1, text_q2])
