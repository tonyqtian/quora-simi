'''
Created on Mar 19, 2017

@author: tonyq
'''

import argparse
from time import sleep

###############################################################################################################################
## Parse arguments
#

parser = argparse.ArgumentParser()
parser.add_argument("--train-path", dest="train_path", type=str, metavar='<str>', default='data/train_sample.csv', help="The path to the training set")
parser.add_argument("--test-path", dest="test_path", type=str, metavar='<str>', default='data/test_sample.csv', help="The path to the test set")
parser.add_argument("--train-feature-path", dest="train_feature_path", type=str, metavar='<str>', default='', help="The path to the train feature set")
parser.add_argument("--test-feature-path", dest="test_feature_path", type=str, metavar='<str>', default='', help="The path to the test feature set")
parser.add_argument("--feature-list", dest="feature_list", type=str, metavar='<str>', default='', help="Feature column list")
parser.add_argument("--feature-idx-start", dest="fidx_start", type=int, metavar='<int>', default=-3, help="Feature index start (default=-3)")
parser.add_argument("--feature-idx-end", dest="fidx_end", type=int, metavar='<int>', default=0, help="Feature index end (default=0)")
parser.add_argument("--out-dir", dest="out_dir_path", type=str, metavar='<str>', default='output', help="The path to the output directory")
parser.add_argument("--optimizer", dest="optimizer", type=str, metavar='<str>', default='adam', help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=rmsprop)")
parser.add_argument("--loss", dest="loss", type=str, metavar='<str>', default='binary_crossentropy', help="Loss function")
parser.add_argument("--activation", dest="activation", type=str, metavar='<str>', default='relu', help="Activation function")
parser.add_argument("--embedding-dim", dest="embd_dim", type=int, metavar='<int>', default=50, help="Embeddings dimension (default=50)")
parser.add_argument("--rnn-dim", dest="rnn_dim", type=int, metavar='<int>', default=4, help="RNN dimension (default=4)")
parser.add_argument("--cnn-dim", dest="cnn_dim", type=int, metavar='<int>', default=0, help="CNN kernels (default=0)")
parser.add_argument("--rnn-layer", dest="rnn_layer", type=int, metavar='<int>', default=1, help="RNN layers (default=1)")
parser.add_argument("--mot-layer", dest="mot_layer", action='store_true', help="Add w2v mot input")
parser.add_argument("--train-batch-size", dest="train_batch_size", type=int, metavar='<int>', default=8, help="Train Batch size (default=8)")
parser.add_argument("--eval-batch-size", dest="eval_batch_size", type=int, metavar='<int>', default=8, help="Eval Batch size (default=8)")
parser.add_argument("--dropout", dest="dropout_prob", type=float, metavar='<float>', default=0.4, help="The dropout probability. To disable, give a negative number (default=0.4)")
parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=1, help="Number of epochs (default=50)")
parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1111, help="Random seed (default=1234)")
parser.add_argument("--plot", dest="plot", action='store_true', help="Save PNG plot")
parser.add_argument("--onscreen", dest="onscreen", action='store_true', help="Show log on stdout")
parser.add_argument("--earlystop", dest="earlystop", type=int, metavar='<int>', default=10, help="Use early stop")
parser.add_argument("--verbose", dest="verbose", type=int, metavar='<int>', default=1, help="Show training process bar during train and val")
parser.add_argument("--valid-split", dest="valid_split", type=float, metavar='<float>', default=0.1, help="Split validation set from training set (default=0.0)")
parser.add_argument("--mem-opt", dest="rnn_opt", type=str, metavar='<str>', default='gpu', help="RNN consume_less (cpu|mem|gpu) (default=gpu)")
parser.add_argument("--eval-on-epoch", dest="eval_on_epoch", action='store_true', help="Test after every epoch")
parser.add_argument("--show-eval-pred", dest="show_evl_pred", type=int, metavar='<int>', default=0, help="Show <num> predicts after every test pred")
parser.add_argument("--w2v-embedding", dest="w2v", type=str, metavar='<str>', default=None, help="Use pre-trained word2vec embedding")
parser.add_argument("--learning-rate", dest="learning_rate", type=float, metavar='<float>', default=0.01, help="Optimizer learning rate")
parser.add_argument("--attention", dest="attention", action='store_true', help="Use Attention Wrapper")
parser.add_argument("--save-model", dest="save_model", action='store_true', help="Save Model Parameters")
parser.add_argument("--model", dest="model", type=str, metavar='<str>', default='rnn', help="Model Type: rnn, doc2vec")
parser.add_argument("--bi-directional", dest="bidirectional", action='store_true', help="Use Bi-directional RNN")
parser.add_argument("--load-model-json", dest="load_model_json", type=str, metavar='<str>', default=None, help="(Optional) Path to the existing model json")
parser.add_argument("--load-model-weights", dest="load_model_weights", type=str, metavar='<str>', default=None, help="(Optional) Path to the existing model weights")
parser.add_argument("--load-vocab-from-file", dest="load_vocab_from_file", type=str, metavar='<str>', default=None, help="(Optional) Path to the existing vocab file")
parser.add_argument("--raw-material", dest="rawMaterial", action='store_true', help="Use Raw Material")
parser.add_argument("--predict-test", dest="predict_test", action='store_true', help="Inference Mode for vector generation")

parser.add_argument("--vec-inference", dest="vecinf", action='store_true', help="Inference Mode for vector generation")
parser.add_argument("--vocab_file", dest="vocab_file", type=str, metavar='<str>', default=None, help="Path to the vocab file")
parser.add_argument("--pbtxt", dest="pbtxt", type=str, metavar='<str>', default=None, help="Path to pb text file")
parser.add_argument("--ckpt", dest="ckpt", type=str, metavar='<str>', default=None, help="Path to check point file")
parser.add_argument("--save_dir", dest="save_dir", type=str, metavar='<str>', default=None, help="Path to saving dump file path")

args = parser.parse_args()

if args.vecinf:
	from src.processing import inference
	inference(args)
else:
	from src.processing import train
	train(args)

print('\a')
sleep(0.3)
print('\a')
sleep(0.3)
print('\a')
sleep(1)
print('\a')
sleep(0.3)
print('\a')
sleep(0.3)
print('\a')
sleep(1)
print('\a')
sleep(0.3)
print('\a')
sleep(0.3)
print('\a')
