# -*- encoding: utf-8 -*-

"""
@version: 0.01
@author: Tony Qiu
@contact: tony.qiu@liulishuo.com
@file: fasttext_test.py
@time: 2018/01/15 午後1:17
"""

# import gensim
import os, sys
# from gensim.models.word2vec import LineSentence
from gensim.models.fasttext import FastText as FT_gensim
from gensim.models.wrappers.fasttext import FastText as FT_wrapper

sys.path.insert(0, '.')
sys.path.insert(0, '..')
# Set FastText home to the path to the FastText executable
ft_home = '/data2/tonyq/fastText/fasttext'

from util.data_processing import get_pdTable, text_cleaner, embdReader

# Set file names for train and test data
data_dir = os.sep + '{}'.format(os.sep).join(['data2', 'tonyq', 'quora-data']) + os.sep
output_dir = os.sep + '{}'.format(os.sep).join(['data2', 'tonyq', 'quora-output']) + os.sep
csv_train_file = data_dir + 'train010.csv'

_, train_question1, train_question2, train_y = get_pdTable(csv_train_file)

train_question1, train_maxLen1 = text_cleaner(train_question1)
train_question2, train_maxLen2 = text_cleaner(train_question2)

train_data = train_question1 + train_question2
# lee_data = LineSentence(lee_train_file)

model_gensim = FT_gensim(size=100)

# build the vocabulary
model_gensim.build_vocab(train_data)

# train the model
print('Training gensim fasttext model...')
model_gensim.train(train_data, total_examples=model_gensim.corpus_count, epochs=model_gensim.iter)
print(model_gensim)

with open(data_dir + 'questions_file.txt', 'w') as fw:
    for line in train_data:
        fw.write(line + '\n')
print('text saved to %s' % (data_dir + 'questions_file.txt'))

# train the model
print('Training wrapper fasttext model...')
model_wrapper = FT_wrapper.train(ft_home, data_dir + 'questions_file.txt')
print(model_wrapper)

# saving a model trained via Gensim's fastText implementation
model_gensim.save(output_dir + 'saved_model_gensim')
loaded_model = FT_gensim.load(output_dir + 'saved_model_gensim')
print(loaded_model)

# saving a model trained via fastText wrapper
model_wrapper.save(output_dir + 'saved_model_wrapper')
loaded_model = FT_wrapper.load(output_dir + 'saved_model_wrapper')
print(loaded_model)

print('night' in model_wrapper.wv.vocab)
print('nights' in model_wrapper.wv.vocab)
print(model_wrapper['night'])
print(model_wrapper['nights'])
model_wrapper.similarity("night", "nights")
model_wrapper.most_similar("nights")

model_wrapper.n_similarity(['sushi', 'shop'], ['japanese', 'restaurant'])

model_wrapper.doesnt_match("breakfast cereal dinner lunch".split())

model_wrapper.most_similar(positive=['baghdad', 'england'], negative=['london'])

# Word Movers distance
sentence_obama = 'Obama speaks to the media in Illinois'.lower().split()
sentence_president = 'The president greets the press in Chicago'.lower().split()

# Remove their stopwords.
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
sentence_obama = [w for w in sentence_obama if w not in stopwords]
sentence_president = [w for w in sentence_president if w not in stopwords]

# Compute WMD.
distance = model_wrapper.wmdistance(sentence_obama, sentence_president)
print(distance)