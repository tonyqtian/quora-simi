'''
Created on Apr 23, 2017

@author: tonyq
'''
from tqdm._tqdm import tqdm
import pickle as pkl

vocabDict = {}
vocabDict['<pad>'] = 0
vocabReverseDict = ['<pad>',]
vocabSet = set([])

idx = 1
with open('../dsk16g/glove.840B.quoraVocab.300d.txt', 'r', encoding='utf8') as fhd:
    with open('../dsk16g/glove.840B.quoraVocab.300d.txt.clean', 'w', encoding='utf8') as fwrt:
        for line in tqdm(fhd):
            wd = line.strip().split(' ')[0]
            if not wd in vocabSet:
                vocabSet.add(wd)
                vocabDict[wd] = idx
                idx += 1
                vocabReverseDict.append(wd)
                fwrt.write(line)
            else:
                print(wd)


with open('quoraVocab.pkl', 'wb') as vocab_file:
    pkl.dump((vocabDict, vocabReverseDict), vocab_file)

print(len(vocabDict))
print(len(vocabReverseDict))
print(vocabReverseDict[:10])
print(vocabReverseDict[-10:])