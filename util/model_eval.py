'''
Created on Mar 18, 2017

@author: tonyq
'''
import logging
from keras.callbacks import Callback
import matplotlib.pyplot as plt
from numpy import around, mean, equal, squeeze

logger = logging.getLogger(__name__)

class Evaluator(Callback):
	
	def __init__(self, args, out_dir, timestr, metric, test_x, test_y, reVocab):
		self.out_dir = out_dir
		self.test_x = test_x
		self.test_y = test_y
		self.best_score = 0
		self.best_epoch = 0
		self.batch_size = args.eval_batch_size
		self.metric = metric
		self.val_metric = 'val_' + metric
		self.timestr = timestr
		self.losses = []
		self.accs = []
		self.val_accs = []
		self.val_losses = []
		self.test_losses = []
		self.test_accs = []
		self.test_precisions = []
		self.plot = args.plot
		self.evl_pred = args.show_evl_pred
		self.save_model = args.save_model
		self.reVocab = reVocab
		
	def eval(self, model, epoch):
		self.test_loss, self.test_metric = model.evaluate(self.test_x, self.test_y, batch_size=self.batch_size)
		self.test_losses.append(self.test_loss)
		self.test_accs.append(self.test_metric)
		if self.evl_pred:
			pred = model.predict(self.test_x, batch_size=self.batch_size)
			preds = around(squeeze(pred))
			precision = mean(equal(self.test_y, preds).astype(int))
			self.test_precisions.append(precision)
			
			self.print_pred(self.test_x[:self.evl_pred], preds[:self.evl_pred], self.test_y[:self.evl_pred])
			self.print_info(epoch, precision)
			
			if self.save_model:
				if precision > self.best_score:
					self.best_score = precision
					self.best_epoch = epoch
					self.model.save_weights(self.out_dir + '/' + self.timestr + 'best_model_weights.h5', overwrite=True)
				self.print_best()

	def on_epoch_end(self, epoch, logs={}):
		self.losses.append(logs.get('loss'))
		self.val_losses.append(logs.get('val_loss'))
		self.accs.append(logs.get(self.metric))
		self.val_accs.append(logs.get(self.val_metric))
		self.eval(self.model, epoch+1)
		if self.plot:
			self.plothem()
		return

	def plothem(self):
		training_epochs = [i+1 for i in range(len(self.losses))]
		plt.plot(training_epochs, self.losses, 'b', label='Train Loss')
		plt.plot(training_epochs, self.accs, 'r.', label='Train Metric')
		plt.plot(training_epochs, self.val_losses, 'g', label='Valid Loss')
		plt.plot(training_epochs, self.val_accs, 'y.', label='Valid Metric')
		plt.legend()
		plt.xlabel('epochs')
		plt.savefig(self.out_dir + '/' + self.timestr + 'LossAccuracy.png')
		plt.close()
		plt.plot(training_epochs, self.test_losses, 'k', label='Test Loss')
		plt.plot(training_epochs, self.test_accs, 'c.', label='Test Metric')
		if self.evl_pred:
			plt.plot(training_epochs, self.test_precisions, 'g.', label='Test Precision')
		plt.legend()
		plt.xlabel('epochs')
		plt.savefig(self.out_dir + '/' + self.timestr + 'TestScore.png')
		plt.close()

	def print_pred(self, infers, preds, reals):
		for (infr, pred, real) in zip(infers, preds, reals):
			infr_line1 = []
			for strin in infr[0]:
				if not strin == 0:
					infr_line1.append(self.reVocab[strin])
			infr_line2 = []
			for strin in infr[1]:
				if not strin == 0:
					infr_line2.append(self.reVocab[strin])
			logger.info('[Test]  ')
			logger.info('[Test]  Line: %s  v.s %s ' % (' '.join(infr_line1), ' '.join(infr_line2)))
			logger.info('[Test]  True: %d  Pred %d ' % (pred, real) )
							
	def print_info(self, epoch, precision):
		logger.info('[Test]  Epoch: %i' % epoch)
		logger.info('[Test]  Precision: %.4f' % precision )

	def print_best(self):
		logger.info('[Test]  Best @ Epoch %i: Score: %.4f' % (self.best_epoch, self.best_score))