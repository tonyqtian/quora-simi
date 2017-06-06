'''
Created on Jun 6, 2017

@author: tonyq
'''
from pandas.io.parsers import read_csv
from pandas.core.frame import DataFrame
from time import strftime

timestr = strftime("%Y%m%d-%H%M%S-")
base = '../output/candi/'

file_list = '20170604-165432-XGB_leaky.clean.csv,' + \
			'20170605-112337-XGB_leaky.csv,' + \
			'20170605-231748-0.1622_lstm_225_120_0.40_0.26.csv,' + \
			'20170606-181044-XGB_leaky.csv,' + \
			'20170606-153627-0.1647_lstm_176_147_0.22_0.23.csv,' + \
			'20170606-183958-0.1693_lstm_191_110_0.23_0.17.csv,' + \
			'20170606-193222-0.1651_lstm_231_100_0.36_0.32.csv,' + \
			'20170606-203300-0.1608_lstm_241_122_0.49_0.30.csv'

test_ids = []
is_duplicate = []

for filename in file_list.split(','):
	print('Processing file ', filename.strip())
	df = read_csv(base + filename.strip())
	if len(test_ids) == 0:
		test_ids = df.test_id
	if len(is_duplicate) == 0:
		is_duplicate = df.is_duplicate
	else:
		is_duplicate += df.is_duplicate
		
is_duplicate /= len(file_list.split(','))
print('Dumping file...')
submission = DataFrame({'test_id':test_ids, 'is_duplicate':is_duplicate})
submission.to_csv(base + timestr + 'ensemble_submit.csv', index=False)
print('Finished')