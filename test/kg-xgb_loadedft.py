'''
Created on Jun 6, 2017

@author: tonyq
'''
import argparse
import functools
from collections import defaultdict
import time
# from util.utils import mkdir

import numpy as np
import pandas as pd
import xgboost as xgb

from nltk.corpus import stopwords
from collections import Counter
from sklearn.metrics import log_loss
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from tqdm._tqdm import tqdm
from pandas.core.frame import DataFrame

# from xgboost import XGBClassifier

def main():
	parser = argparse.ArgumentParser(description='XGB with Handcrafted Features')
	parser.add_argument('--save', type=str, default='XGB_leaky', help='save_file_names')
	parser.add_argument('--train_features', type=str)
	parser.add_argument('--test_features', type=str)
	parser.add_argument('--train_extra', type=str, default='')
	parser.add_argument('--test_extra', type=str, default='')
	parser.add_argument('--feature_list', type=str, default='')
	parser.add_argument('--extra_feature_list', type=str, default='')
	args = parser.parse_args()
	timestr = time.strftime("%Y%m%d-%H%M%S-")
	output_dir = '../output/' + time.strftime("%m%d")
	
	print("Reading train features...")
	df_train = pd.read_csv(args.train_features, encoding="ISO-8859-1")
	train_features = DataFrame()
	feature_list = args.feature_list.split(',')
	for feature_name in feature_list:
		train_features[feature_name.strip()] = df_train[feature_name.strip()]
	
	y_train = df_train['is_duplicate'].values
		
	if args.train_extra is not '':
		print("Reading train 1bowl features...")
		df_train = pd.read_csv(args.train_extra, encoding="ISO-8859-1")
		extra_feature_list = args.extra_feature_list.split(',')
		for feature_name in extra_feature_list:
			train_features[feature_name.strip()] = df_train[feature_name.strip()]
	del df_train
	print('Final train columns', train_features.columns)
	print('Shape: ', train_features.shape)
	
	X_train, X_valid, y_train, y_valid = train_test_split(train_features, y_train, test_size=0.06, random_state=4242)
	del train_features
	
	#UPDownSampling
	print("Train Sampling...")
	pos_train = X_train[y_train == 1]
	neg_train = X_train[y_train == 0]
	X_train = pd.concat((neg_train, pos_train.iloc[:int(0.8*len(pos_train))], neg_train))
	y_train = np.array([0] * neg_train.shape[0] + [1] * pos_train.iloc[:int(0.8*len(pos_train))].shape[0] + [0] * neg_train.shape[0])
	print(np.mean(y_train))
	del pos_train, neg_train

	print("Valid Sampling...")
	pos_valid = X_valid[y_valid == 1]
	neg_valid = X_valid[y_valid == 0]
	X_valid = pd.concat((neg_valid, pos_valid.iloc[:int(0.8 * len(pos_valid))], neg_valid))
	y_valid = np.array([0] * neg_valid.shape[0] + [1] * pos_valid.iloc[:int(0.8 * len(pos_valid))].shape[0] + [0] * neg_valid.shape[0])
	print(np.mean(y_valid))
	del pos_valid, neg_valid
	
	params = {}
	params['objective'] = 'binary:logistic'
	params['eval_metric'] = 'logloss'
	params['eta'] = 0.02
	params['max_depth'] = 7
	params['subsample'] = 0.6
	params['base_score'] = 0.2
	# params['scale_pos_weight'] = 0.2

	print("DMatrix...")
	d_train = xgb.DMatrix(X_train, label=y_train)
	d_valid = xgb.DMatrix(X_valid, label=y_valid)

	watchlist = [(d_train, 'train'), (d_valid, 'valid')]

	print("XGBoost training...")
	bst = xgb.train(params, d_train, 2500, watchlist, early_stopping_rounds=50, verbose_eval=50)
	print(log_loss(y_valid, bst.predict(d_valid)))
	bst.save_model(output_dir + '/' + timestr + args.save + '.mdl')
	
	print('Loading Test Features')
	df_test = pd.read_csv(args.test_features, encoding="ISO-8859-1")
	test_features = DataFrame()
	feature_list = args.feature_list.split(',')
	for feature_name in feature_list:
		test_features[feature_name.strip()] = df_test[feature_name.strip()]
		
	if args.test_extra is not '':
		print("Reading test 1bowl features...")
		df_test = pd.read_csv(args.test_extra, encoding="ISO-8859-1")
		extra_feature_list = args.extra_feature_list.split(',')
		for feature_name in extra_feature_list:
			test_features[feature_name.strip()] = df_test[feature_name.strip()]
	
	print('Final test columns ', test_features.columns)
	print('Shape: ', test_features.shape)

	d_test = xgb.DMatrix(test_features)
	p_test = bst.predict(d_test)
	sub = pd.DataFrame()
	sub['test_id'] = df_test['test_id']
	sub['is_duplicate'] = p_test
	sub.to_csv(output_dir + '/' + timestr + args.save + '.csv', index=False)
	print('Dumped inference to file '+ timestr + args.save + '.csv')
	print('Finished.')
	
if __name__ == '__main__':
	main()