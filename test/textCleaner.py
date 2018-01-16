'''
Created on Apr 23, 2017

@author: tonyq
'''

import pandas as pd
from nltk.tokenize import word_tokenize
from tqdm._tqdm import tqdm
import csv
import re
from keras.preprocessing.text import Tokenizer

MAX_NB_WORDS = 200000


def get_words(text):
# 	word_split = re.compile('[^a-zA-Z0-9_\\+\\-]')
# 	return [word.strip().lower() for word in word_split.split(text)]
    text = str(text)
# 	text = text.replace('’s', ' ’s')
    text = text.replace('…', ' ')
    text = text.replace('”', ' ')
    text = text.replace('“', ' ')
    text = text.replace('‘', ' ')
    text = text.replace('’', ' ')
    text = text.replace('"', ' ')
# 	text = text.replace("'", " ")
    text = text.replace('-', ' ')
# 	text = text.replace('/', ' ')
    text = text.replace("\\", " ")
    # Clean the text
    # text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    # text = re.sub(r"\^", " ^ ", text)
    # text = re.sub(r"\+", " + ", text)
    # text = re.sub(r"\-", " - ", text)
    # text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    # text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text


# test = pd.read_csv("test.csv")
# totalen = len(test.question1)
# print('Test size: ', totalen)

train = pd.read_csv("train.csv")
totalen = len(train.question1)
print('Train size: ', totalen)

# text_material = test.question1 + test.question2 + train.question1 + train.question2
text_material = train.question2[:1000]

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True)
tokenizer.fit_on_texts(text_material)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))
print(len(tokenizer.word_counts))
print(tokenizer.word_counts)

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
