'''
Created on Mar 17, 2017

@author: tonyq
'''
import codecs
import csv

import pandas as pd
import numpy as np
import re, sys, os
import time
from bs4 import BeautifulSoup
import logging
from keras.preprocessing.sequence import pad_sequences
from tqdm._tqdm import tqdm

from numpy import array, zeros
import operator

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

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
# 	text = text.replace('’s', ' ’s')
# 	text = text.replace('…', ' ')
# 	text = text.replace('”', ' ')
# 	text = text.replace('“', ' ')
# 	text = text.replace('‘', ' ')
# 	text = text.replace('’', ' ')
# 	text = text.replace('"', ' ')
# 	text = text.replace("'", " ")
# 	text = text.replace('-', ' ')
# 	text = text.replace('/', ' ')
# 	text = text.replace("\\", " ")
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
    # text = text.replace(" e g ", " eg ")
    # text = text.replace(" b g ", " bg ")
    # text = text.replace(" u s ", " american ")
    # text = re.sub(r"\0s", "0", text)
    text = text.replace(" 9 11 ", "911")
    text = text.replace("e - mail", "email")
    # text = text.replace("j k", "jk")
    text = re.sub(r"\s{2,}", " ", text)
    # text = text.replace("\\", " ")
    text = text.replace('"', ' ')

    return text.strip()
# 	return word_tokenize(text)


# The function "text_to_wordlist" is from
# https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text
def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.

    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]

    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
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
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    # Return a list of words
    return (text)


def csv_processing(path, test=False):
    texts_1 = []
    texts_2 = []
    labels = []
    test_ids = []
    with codecs.open(path, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        if test == False:
            for values in tqdm(reader):
                texts_1.append(text_to_wordlist(values[3]))
                texts_2.append(text_to_wordlist(values[4]))
                labels.append(int(values[5]))
            return texts_1, texts_2, labels
        else:
            for values in tqdm(reader):
                texts_1.append(text_to_wordlist(values[1]))
                texts_2.append(text_to_wordlist(values[2]))
                test_ids.append(values[0])
            return texts_1, texts_2, test_ids


def get_pdTable(path, notag=False):
    logger.info(' Processing pandas csv ')
    pdtable = pd.read_csv(path)
    if notag:
        try:
            return pdtable.test_id, pdtable.question1, pdtable.question2
        except AttributeError:
            return pdtable.id, pdtable.question1, pdtable.question2
    else:
        return pdtable.id, pdtable.question1, pdtable.question2, pdtable.is_duplicate


def text_cleaner(table):
    textTable = []
    maxLen = 0
    for text in tqdm(table, file=sys.stdout):
        text = get_words(text)
        textTable.append(text)
        if len(text) > maxLen:
            maxLen = len(text)
    return textTable, maxLen


def tokenizeIt(table, clean=False, addHead=None):
    tokenizedTable = []
    maxLen = 0
    for text in tqdm(table, file=sys.stdout):
        if clean:
# 			text = stripTagsAndUris(text)
            text = word_tokenize(get_words(text))
            if not addHead is None:
                text = [addHead] + text
            tokenizedTable.append(text)
            if len(text) > maxLen:
                maxLen = len(text)
        else:
            text = str(text).split(' ')
            if not addHead is None:
                text = [addHead] + text
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


def embdReader(embd_path, embd_dim, word_index, max_nb_words, fasttext_source='', ft_dim=0,
               ft_home='/data2/tonyq/fastText/fasttext', output_dir='/data2/tonyq/quora-output/', skip_header=False,
               initializer='glorot'):
    ########################################
    ## index word vectors
    ########################################
    if not embd_path == '':
        logger.info('Indexing word vectors...')
        embeddings_index = {}
        with open(embd_path, 'r', encoding='utf8') as f:
            if skip_header or embd_path.endswith('.vec'):
                next(f)
            for line in tqdm(f):
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        logger.info('Found %d word vectors in embedding file.' % len(embeddings_index))

    ########################################
    ## prepare fasttext
    ########################################
    if not fasttext_source == '':
        from gensim.models.wrappers.fasttext import FastText as FT_wrapper
        if fasttext_source.endswith('.bin'):
            loaded_model = FT_wrapper.load(fasttext_source)
            print(loaded_model)
        else:
            _, train_question1, train_question2 = get_pdTable(fasttext_source, notag=True)
            train_question1, train_maxLen1 = text_cleaner(train_question1)
            train_question2, train_maxLen2 = text_cleaner(train_question2)
            train_data = train_question1 + train_question2
            print('Train data lines %d' % len(train_data))

            with open(output_dir + 'questions_file.txt', 'w') as fw:
                for line in train_data:
                    fw.write(line + '\n')
            print('Text saved to %s' % (output_dir + 'questions_file.txt'))

            # train the model
            print('Training wrapper fasttext model...')
            tstart = time.time()
            model_wrapper = FT_wrapper.train(ft_home, output_dir + 'questions_file.txt', size=ft_dim)
            tend = time.time()
            print('Time elapsed for training wrapper model %.2f' % (tend - tstart))
            print(model_wrapper)

            # saving a model trained via fastText wrapper
            print('Loading fasttext wrapper model...')
            model_wrapper.save(output_dir + 'saved_model_wrapper.bin')

    ########################################
    ## prepare embeddings
    ########################################
    logger.info('Preparing embedding matrix based on given word list...')
    nb_words = min(max_nb_words, len(word_index))+1

    w2v_oov = 0
    ft_oov = []
    if initializer == 'zero':
        # zero initialization of embedding matrix
        embedding_matrix = np.zeros((nb_words, embd_dim+ft_dim))
    elif initializer == 'glorot':
        # glorot uniform initialization of embedding matrix
        scale = 1 / nb_words              # fan_in
        # scale = 1 / (embd_dim + ft_dim)   # fan_out
        limit = np.sqrt(3. * scale)
        embedding_matrix = np.random.uniform(low=-limit, high=limit, size=(nb_words, embd_dim+ft_dim))
    else:
        raise NotImplementedError

    reverseDict = ['']*nb_words
    for word, i in tqdm(word_index.items()):
        if not embd_path == '':
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i][:embd_dim] = embedding_vector
                reverseDict[i] = word
            else:
                reverseDict[i] = '<' + word + '>'
                w2v_oov += 1
        if not fasttext_source == '':
            try:
                embedding_matrix[i][embd_dim:] = model_wrapper[word]
                reverseDict[i] = word
            except KeyError:
                reverseDict[i] = '<' + word + '>'
                ft_oov.append(word)

    logger.info('Word embeddings shape: %r (%d+%d)' % (embedding_matrix.shape, embd_dim, ft_dim))
    if not embd_path == '':
        logger.info('Word2Vec null embeddings: %d' % w2v_oov)
    if not fasttext_source == '':
        logger.info('FastText null embeddings: %d' % len(ft_oov))
        logger.info('FastText OOV: %r' % ft_oov)
    return embedding_matrix, reverseDict


def w2vEmbdReader(embd_path, reVocab, embd_dim):
    logger.info('  getting pre-trained embedding from file... ')
    logger.info('  embedding length: %i  dim: %i  ' % (len(reVocab), embd_dim))
    embd_matrix = np.zeros( (len(reVocab), embd_dim) )
    with open(embd_path, 'r', encoding='utf8') as fhd:
        idx = 1 # let 1st padding line all zeros
        for line in tqdm(fhd, total=len(reVocab)):
            elem = line.strip().split(' ')
            assert len(elem) == embd_dim + 1, 'Incorrect Embedding Dimension, expect %d but got %d ' % (embd_dim, len(elem)-1)
            w2vec = np.asarray(elem[1:], dtype='float32')
            embd_matrix[idx] = w2vec
            idx += 1
    return embd_matrix
