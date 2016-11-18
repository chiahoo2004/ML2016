from __future__ import absolute_import
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
from keras.models import model_from_json
import numpy as np
import pickle
import pandas as pd
import sys
import keras
from keras import backend as K
K.set_image_dim_ordering('tf')

import sys
path = sys.argv[1]
input_model = sys.argv[2]
prediction = sys.argv[3]
size=3*32*32
nb_classes = 10
nb_hidden_layers = [size, 2000, 1000, 500]

test = pickle.load( open(path+'test.p','rb') )
X_test = np.array( pd.Series(test).values[1] )
X_test = X_test.astype("float32") / 255.0

'''
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
'''
#model = keras.models.load_model(input_model)
#model = pickle.load( open( input_model, 'rb' ) )


model = Sequential()
model.add(Dense(input_dim=nb_hidden_layers[0],output_dim=nb_hidden_layers[1]))
model.add(Dense(1000))
model.add(Dense(500))
model.add(Dense(nb_classes, activation='softmax'))
'''
model.add(Activation('sigmoid'))
model.add(Dense(689))
model.add(Activation('sigmoid'))
model.add(Dense(689))
model.add(Activation('sigmoid'))
model.add(Dense(10))
'''

model.load_weights(input_model)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])






Y_test_pred = model.predict(X_test)
y_test_pred = np.argmax(Y_test_pred, axis=1)






title = np.array([['ID','class']])        
ids = np.array( np.arange(10000) )       
ids = ids.astype(np.int)
y_test_pred = y_test_pred.astype(np.int)       
result = np.c_[ids,y_test_pred]
result = np.r_[title,result]
np.savetxt(prediction, result, delimiter=",", fmt="%s")


