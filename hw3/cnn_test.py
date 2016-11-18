from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import keras
from keras import backend as K
K.set_image_dim_ordering('tf')

import sys
path = sys.argv[1]
input_model = sys.argv[2]
prediction = sys.argv[3]

nb_classes = 10

test = pickle.load( open(path+'test.p','rb') )
X_test = np.array( pd.Series(test).values[1] )

X_test_reshape = X_test.reshape((10000, 3, 32, 32))

tmp2 = np.zeros((10000,32,32,3))
for i in range(10000):
#    if(i%1000==0):
#        print '#',i
    stack = np.dstack((X_test_reshape[i][1],X_test_reshape[i][2],X_test_reshape[i][0]))
    tmp2[i] = np.copy(stack)

X_test = np.copy(tmp2)
X_test = X_test.astype('float32')
X_test /= 255





model = keras.models.load_model(input_model)







Y_test_pred = model.predict(X_test)
y_test_pred = np.argmax(Y_test_pred, axis=1)



   
title = np.array([['ID','class']])        
ids = np.array( np.arange(10000) )       
ids = ids.astype(np.int)
y_test_pred = y_test_pred.astype(np.int)       
result = np.c_[ids,y_test_pred]
result = np.r_[title,result]
np.savetxt(prediction, result, delimiter=",", fmt="%s")