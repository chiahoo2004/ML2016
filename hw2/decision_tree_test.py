import numpy as np
import csv

import sys
import pickle

class Node(object):
    def __init__(self, seq=0):
        self.seq = seq
        self.left = 0
        self.right = 0
        self.dimension = 0
        self.threshold = 0
        self.result = 0

def classify(X_test):
    
    global tree
    
    n=0
    while tree[n].result==-1:
        if( X_test[ tree[n].dimension ] <= tree[n].threshold ):
            n = tree[n].left
        else:
            n = tree[n].right
    return tree[n].result

def validate(X_test, y_test):   
    
    num = y_test.shape[0]
#    num = 4001   ################################### 
    err = 0
    for i in range(num):
        pred = classify(X_test[i])
        if( pred != y_test[i] ):
           err += 1
    acc = 1 - float(err)/num
    return acc

if __name__ == '__main__':

    model_name = sys.argv[1]
    testing_data = sys.argv[2]
    prediction = sys.argv[3]
    
    with open(testing_data, 'rb') as f_:
        reader_ = csv.reader(f_)
        raw_ = list(reader_)  
    
      
    raw_ = np.array(raw_)
    raw_ = raw_.astype(np.float)
    test = np.delete(raw_, 0, 1)          
    
    
    tree = np.load(model_name) 
#    tree = pickle.load( open( 'model.pkl', 'rb' ) )
    
    test_num = test.shape[0]
    pred = np.zeros(test_num)
    for i in range(test_num):
        pred[i] = classify(test[i])
    
        
    title = np.array([['id','label']])        
    ids = raw_[:,0]      
    ids = ids.astype(np.int)
    pred = pred.astype(np.int)
    result = np.c_[ids,pred]
    result = np.r_[title,result]
    np.savetxt(prediction, result, delimiter=",", fmt="%s")

