'''
Created on Jun 1, 2017

@author: tonyq
'''
import pandas as pd
from tqdm._tqdm import tqdm
import csv


train = pd.read_csv("../output/0604/20170604-165432-XGB_leaky.csv")
totalen = len(train.is_duplicate)
print('Total size: ', totalen)

fulllist = zip(train.test_id, train.is_duplicate)
length = len(train.is_duplicate)
del train

with open("../output/0604/20170604-165432-XGB_leaky.clean.csv", "w", encoding='utf8') as fwrt:
	writer_sub = csv.writer(fwrt)
	writer_sub.writerow(['test_id','is_duplicate'])
	for (theid, dup) in tqdm(fulllist, total=length):
		writer_sub.writerow([theid, dup])
