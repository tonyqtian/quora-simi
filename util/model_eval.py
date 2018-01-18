'''
Created on Mar 18, 2017

@author: tonyq
'''
import logging
import six
from collections import OrderedDict, Iterable
from keras.callbacks import Callback
import matplotlib.pyplot as plt
from numpy import around, mean, equal, squeeze, ndarray
from random import randint

logger = logging.getLogger(__name__)

judgeInfo = {False:'Incorect', True:''}

class PlotPic(Callback):
    def __init__(self, args, out_dir, timestr, metric):
        self.out_dir = out_dir
        self.metric = metric
        self.val_metric = 'val_' + metric
        self.timestr = timestr
        self.losses = []
        self.accs = []
        self.val_accs = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.accs.append(logs.get(self.metric))
        self.val_accs.append(logs.get(self.val_metric))
        self.plothem()
        return

    def plothem(self):
        training_epochs = [i+1 for i in range(len(self.losses))]

        plt.plot(training_epochs, self.losses, 'b', label='Train Loss')
        plt.plot(training_epochs, self.val_losses, 'g', label='Valid Loss')
        plt.legend()
        plt.xlabel('epochs')
        plt.savefig(self.out_dir + '/' + self.timestr + 'TrainLoss.png')
        plt.close()

        plt.plot(training_epochs, self.accs, 'r.', label='Train Metric')
        plt.plot(training_epochs, self.val_accs, 'y.', label='Valid Metric')
        plt.legend()
        plt.xlabel('epochs')
        plt.savefig(self.out_dir + '/' + self.timestr + 'TrainAcc.png')
        plt.close()


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
        self.print_info(epoch, self.test_loss, self.test_metric)
        if self.evl_pred:
            pred = model.predict(self.test_x, batch_size=self.batch_size)
            preds = squeeze(pred)
            precision = mean(equal(self.test_y, around(preds)).astype(int))
            self.test_precisions.append(precision)

            idx = randint(0, len(self.test_y)//2 )
            self.print_pred([ self.test_x[0][idx:idx+self.evl_pred], self.test_x[1][idx:idx+self.evl_pred] ],
                         preds[idx:idx+self.evl_pred], self.test_y[idx:idx+self.evl_pred])

# 			if self.save_model:
# 				if self.test_loss < self.best_score:
# 					self.best_score = self.test_loss
# 					self.best_epoch = epoch
# 					self.model.save_weights(self.out_dir + '/' + self.timestr + 'best_model_weights.h5', overwrite=True)
# 				self.print_best()

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
        plt.plot(training_epochs, self.val_losses, 'g', label='Valid Loss')
        plt.plot(training_epochs, self.test_losses, 'k', label='Test Loss')
        plt.legend()
        plt.xlabel('epochs')
        plt.savefig(self.out_dir + '/' + self.timestr + 'TrainTestLoss.png')
        plt.close()

        plt.plot(training_epochs, self.accs, 'r.', label='Train Metric')
        plt.plot(training_epochs, self.val_accs, 'y.', label='Valid Metric')
        plt.plot(training_epochs, self.test_accs, 'c.', label='Test Metric')
        plt.legend()
        plt.xlabel('epochs')
        plt.savefig(self.out_dir + '/' + self.timestr + 'TrainTestAcc.png')
        plt.close()

    def print_pred(self, infers, preds, reals):
        for (infr0, infr1, pred, real) in zip(infers[0], infers[1], preds, reals):
            infr_line1 = []
            for strin in infr0:
                if not strin == 0:
                    infr_line1.append(self.reVocab[int(strin)])
            infr_line2 = []
            for strin in infr1:
                if not strin == 0:
                    infr_line2.append(self.reVocab[int(strin)])
            logger.info('[Test]  ')
            logger.info('[Test]  Q1:  %s ' % ' '.join(infr_line1))
            logger.info('[Test]  Q2:  %s ' % ' '.join(infr_line2))
            try:
                logger.info('[Test]  True: %d  Pred %d (%.4f) %s ' %
                        (real, around(pred), pred, judgeInfo[real==around(pred)]) )
            except ValueError:
                logger.info('[Test]  True: %d  Pred %f ' % (real, pred) )

    def print_info(self, epoch, logloss, acc):
        logger.info('[Test]  Epoch: %i  Test Loss: %.4f  Test Accuracy: %.4f%%' % (epoch, logloss, 100*acc))
        logger.info('[Test]  ----------------------------------------------------- ')

    def print_best(self):
        logger.info('[Test]  ')
        logger.info('[Test]  Best @ Epoch %i: Log Loss: %.4f' % (self.best_epoch, self.best_score))
        logger.info('[Test]  ')


class TrainLogger(Callback):
    """Callback that streams epoch results to a csv file.

    Supports all values that can be represented as a string,
    including 1D iterables such as np.ndarray.

    # Example
        ```python
        csv_logger = CSVLogger('training.log')
        model.fit(X_train, Y_train, callbacks=[csv_logger])
        ```

    # Arguments
        filename: filename of the csv file, e.g. 'run/log.csv'.
        separator: string used to separate elements in the csv file.
        append: True: append if file exists (useful for continuing
            training). False: overwrite existing file,
    """

    def __init__(self):
        self.logger = logger
        self.keys = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, ndarray) and k.ndim == 0
            if isinstance(k, six.string_types):
                return k
            elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict([(k, logs[k]) if k in logs else (k, 'NA') for k in self.keys])

        self.keys = sorted(logs.keys())

        row_dict = OrderedDict({'epoch': epoch})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        outputlist = []
        for ky, vl in row_dict.items():
            if ky == 'epoch':
                outputlist.append("%s %d" % (ky, vl))
            elif ky.endswith('acc'):
                outputlist.append("%s %.2f%%" % (ky, vl*100))
            else:
                outputlist.append("%s %.4f" % (ky, vl))
        self.logger.info("\n" + " | ".join(outputlist))