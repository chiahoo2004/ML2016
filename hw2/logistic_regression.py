import numpy as np
import csv

import sys
training_data = sys.argv[1]
output_model = sys.argv[2]

       
with open(training_data, 'rb') as f:
    reader = csv.reader(f)
    raw = list(reader)

raw = np.array(raw)
raw = raw.astype(np.float) 
ans = raw[:,-1:]
train = np.delete(raw, 0, 1)
train = np.delete(train, -1, 1) 



feature_num = train.shape[1]
#feature_num = 57
     
#%%
rate = 8e-3
iter_num = 2650
lamb = 0
power_num = 1

w_num = feature_num*power_num

b = 0
b_table = np.zeros(iter_num)
w = np.ones(w_num)




entry = np.ones( w_num )

ada_b = 0
ada_g = np.zeros(w_num)

sample_num = train.shape[0]
grad_w = np.zeros(w_num)

batch = 5

w = np.random.randn(train.shape[1]*power_num) / train.shape[1] / train.shape[0] / power_num
w = np.resize(w,(train.shape[1]*power_num,1))
#w = np.copy(w_test)

batch_counter = 0

for ite in range(iter_num):
    
    idx = [] 
    
    random = np.random.permutation(sample_num)

    for n in random:
        
               
        idx.append(n)
        batch_counter += 1
        
        if(batch_counter==batch):
        
            train_this = train[idx]
            ans_this = ans[idx]
            
            
            grad_b = 0
            grad_w = np.zeros(w_num)
    
            y = np.zeros([len(idx),1])
            y += b
            for power in range(1,power_num+1):
                w_this = w[0+ feature_num *(power-1): feature_num + feature_num *(power-1)]
                entry_this = entry[0+ feature_num *(power-1): feature_num + feature_num *(power-1)]                   
                prod = (train_this**power).dot(w_this)
                
                y += np.sum( prod,axis=1 ).reshape( len(idx) ,1)
                
                
            sigmoid = 1. / (1+np.exp(-y))
            temp = ( sigmoid-ans_this )       
            
            grad_b = np.sum(temp)
            for power in range(1,power_num+1):
                w_this = w[0+ feature_num *(power-1): feature_num + feature_num *(power-1)]
                entry_this = entry[0+ feature_num *(power-1): feature_num + feature_num *(power-1)]
                                   
                norm = (train_this**power).T.dot(temp)   
                
                grad_w[0+ feature_num *(power-1): feature_num + feature_num *(power-1)] = norm.reshape(feature_num,)     
        
            ada_b += grad_b**2
            for i in range(w_num):
                if(entry[i]==0):
                    ada_g[i] = 0
                else:
                    ada_g[i] += grad_w[i]**2
    
            grad_w[ada_g==0] = 0
            if(ada_b==0):
                grad_b = 0
    
            if(ada_b!=0):
                b = b - rate * grad_b / np.sqrt(ada_b)
            for i in range(w_num):
                if(ada_g[i]!=0):
                    w[i] = w[i] - rate * grad_w[i] / np.sqrt(ada_g[i])
                    
            dummy = 0
            batch_counter = 0
            idx = []

        
            
    product = train.dot(w) + b
    temp = 1+np.exp(-product)
    sig = 1. / temp
    pred = np.copy(sig)
    pred[sig >= 0.5] = 1
    pred[sig < 0.5] = 0
    err = np.sum( abs(pred-ans) ) /np.shape(pred)[0]
    acc_train = 1-err
    
    
    
    
    
    print('\tepoch: %d; train: %f' % (ite, acc_train))

model = np.append(w,b)    
np.save(output_model, model)
    


