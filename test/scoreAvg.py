'''
Created on Jun 6, 2017

@author: tonyq
'''
from pandas.io.parsers import read_csv
from pandas.core.frame import DataFrame

base = '../output/candi/'
file_list = '20170604-165432-XGB_leaky.clean.csv,' + \
			'20170605-112337-XGB_leaky.csv,' + \
			'20170605-231748-0.1622_lstm_225_120_0.40_0.26.csv' + \
			' 20170606-181044-XGB_leaky.csv'

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
submission.to_csv(base + 'final_submit.csv', index=False)
print('Finished')