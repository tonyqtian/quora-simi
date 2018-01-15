'''
Created on Mar 23, 2017

@author: tonyq
'''
import keras.backend as K

def my_binary_crossentropy(y_true, y_pred):
    return K.mean(cal_binary_crossentropy(y_pred, y_true))

def cal_binary_crossentropy(y_true, y_pred):
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    return - K.sum(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred), axis=-1)
