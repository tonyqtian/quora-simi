'''
Created on Jun 4, 2017

@author: tonyq
'''
from pandas import read_csv
import numpy as np

# df_train = read_csv('D:\\BaiduCloud\\PostGraduate\\DL\\Quora\\train_extra_features.csv', encoding="ISO-8859-1")
# df_train = read_csv('../data/train_extra_features_sample.csv', encoding="ISO-8859-1")
# df_train = read_csv('D:\\BaiduCloud\\PostGraduate\\DL\\Quora\\train_features_1bowl.csv', encoding="ISO-8859-1")
df_train = read_csv('../data/train_features_1bowl_sample.csv', encoding="ISO-8859-1")
df_train = df_train.drop('id', axis=1)
# df_train = df_train.drop('test_id', axis=1)
df_train = df_train.drop('qid1', axis=1)
df_train = df_train.drop('qid2', axis=1)
df_train = df_train.drop('question1', axis=1)
df_train = df_train.drop('question2', axis=1)
df_train = df_train.drop('is_duplicate', axis=1)
df_train = df_train.drop('question1_nouns', axis=1)
df_train = df_train.drop('question2_nouns', axis=1)

df_train = df_train.replace([np.inf, -np.inf, np.nan], 0)

print('Total feature count: ', len(df_train.columns))
sigFeat = []
countFeat = []
unboundFeat = []
for itm in df_train.columns:
    print('Feature: ', itm)
    Min = min(df_train[itm])
    Max = max(df_train[itm])
    Avg = np.average(df_train[itm])
    if Min >= -1.0 and Max <= 2.0:
        sigFeat.append(itm)
    elif Min >= 0.0 and Max == np.around(Max):
        countFeat.append(itm)
    else:
        unboundFeat.append(itm)

print('Features within [-1, 1]', ','.join(sigFeat))
for itm in sigFeat:
    print('  Feature: ', itm)
    print('    Min: %.4f' % min(df_train[itm]))
    print('    Max: %.4f' % max(df_train[itm]))
    print('    Avg: %.4f' % np.average(df_train[itm]))

print('Features countable', ','.join(countFeat))
for itm in countFeat:
    print('  Feature: ', itm)
    print('    Min: ', min(df_train[itm]))
    print('    Max: ', max(df_train[itm]))
    print('    Avg: ', np.average(df_train[itm]))

print('Features unbounded', ','.join(unboundFeat))
for itm in unboundFeat:
    print('  Feature: ', itm)
    print('    Min: %.4f' % min(df_train[itm]))
    print('    Max: %.4f' % max(df_train[itm]))
    print('    Avg: %.4f' % np.average(df_train[itm]))

# print(df_train['jaccard_distance'])
# df_train = df_train.replace([np.inf, -np.inf], 0).replace(np.nan, 0)
# print(df_train['wmd'])
# print(df_train['norm_wmd'])
# print(df_train['kur_q1vec'])
# print(df_train['word_match'])
# print(df_train['tfidf_wm'])
print(df_train['z_tfidf_mean2'])