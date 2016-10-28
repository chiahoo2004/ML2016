import numpy as np
import csv

import sys
model_name = sys.argv[1]
testing_data = sys.argv[2]
prediction = sys.argv[3]

with open(testing_data, 'rb') as f_:
    reader_ = csv.reader(f_)
    raw_ = list(reader_)  
     
  
raw_ = np.array(raw_)
raw_ = raw_.astype(np.float)
test = np.delete(raw_, 0, 1)          


model = np.load(model_name) 
w = model[0:-1]
w = w.reshape(w.shape[0],1)
b = model[-1]

test_num = test.shape[0]
pred = np.zeros(test_num)

for n in range(test_num):
    
    product = test.dot(w) + b
    temp = 1+np.exp(-product)
    sig = 1. / temp
    pred = np.copy(sig)
    pred[sig >= 0.5] = 1
    pred[sig < 0.5] = 0
 
       
title = np.array([['id','label']])        
ids = raw_[:,0]        
ids = ids.astype(np.int)
pred = pred.astype(np.int)       
result = np.c_[ids,pred]
result = np.r_[title,result]
np.savetxt(prediction, result, delimiter=",", fmt="%s")