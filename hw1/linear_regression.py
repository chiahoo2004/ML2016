import numpy as np
import csv
   

entry1 = np.array([
                    0,  #'AMB_TEMP':   
                    1,  #'CH4':        
                    0,  #'CO':         
                    0,  #'NMHC':       
                    1,  #'NO':         
                    1,  #'NO2':        
                    1,  #'NOx':        
                    0,  #'O3':         
                    0,  #'PM10':
                    1,  #'PM2.5'
                    0,  #'RAINFALL'
                    0,  #'RH':         
                    1,  #'SO2':        
                    1,  #'THC':        
                    0,  #'WD_HR':      
                    0,  #'WIND_DIREC': 
                    0,  #'WIND_SPEED': 
                    0   #'WS_HR': 
                ])

entry2 = np.array([
                    0,  #'AMB_TEMP':   
                    0,  #'CH4':        
                    0,  #'CO':         
                    0,  #'NMHC':       
                    0,  #'NO':         
                    0,  #'NO2':        
                    0,  #'NOx':        
                    0,  #'O3':         
                    0,  #'PM10':
                    1,  #'PM2.5'
                    0,  #'RAINFALL'
                    0,  #'RH':         
                    0,  #'SO2':        
                    0,  #'THC':        
                    0,  #'WD_HR':      
                    0,  #'WIND_DIREC': 
                    0,  #'WIND_SPEED': 
                    0   #'WS_HR': 
                ])

zero = np.zeros(18)
zeros = np.tile(zero,2)
entry1 = np.tile(entry1,9)
entry2 = np.tile(entry2,9)

entry = np.append(entry1,entry2)
entry = np.append(zeros,entry)
       
with open('train.csv', 'rb') as f:
    reader = csv.reader(f)
    raw = list(reader)

raw = np.array(raw)
raw[raw=='NR']=0


#%%

train = []
ans = []

for i in range(240):
    for j in range(3,27):
        if(i==0):
            if(j<12):
                pass
            elif(j==12):
                ans = np.array(raw[10,12])
            else:
                ans = np.append(ans,raw[10,j])
        else:   
            ans = np.append(ans,raw[10+18*i,j])


for i in range(240):
    for j in range(3,27):
        if(i==0 and j==3):
            train = raw[1:19,3]
        else:
            new = raw[1+18*i:19+18*i,j]
            train = np.append(train,new)   
'''
attr = np.zeros([18,5760])
for i in range(4320):
    attr[i%18,0+(i/18)*24:5784] = raw[i+1,3:27]
         
attr_test = 
'''
            
train = train.astype(np.float) 
ans = ans.astype(np.float)   

    
     
#%%
#rate = 0.5 * 10**(-13) -6          7 * 10**(-7)
rate = 0.1
iter_num = 5000

lamb = 0 #

power_num = 1
w_num = 18*9*power_num

b = 0
b_table = np.zeros(iter_num)
w = np.zeros(w_num)
w_table = np.zeros([w_num,iter_num])


rmse_table = np.zeros(iter_num)

ada_b = 0
ada_g = np.zeros(w_num)

sample_num = 5751   # 239 639  5751



for ite in range(iter_num):
    
    print ite-1, rmse_table[ite-1]
    
    grad_b = 0
    grad_w = np.zeros(w_num)
    rmse = 0
    
    for n in range(sample_num):
        
        train_this = train[0+18*n:162+18*n] # [0+18*n:162+18*n] 18 162 432
        ans_this = train[171+18*n]
        
        y = b
        for power in range(1,power_num+1):
            w_this = w[0+162*(power-1):162+162*(power-1)]
            entry_this = entry[0+162*(power-1):162+162*(power-1)]
            y += np.sum( entry_this * w_this * (train_this**power) )
       
        temp = 2*( ans_this-y ) * ( -1 )
        grad_b += temp
        
        for power in range(1,power_num+1):
            w_this = w[0+162*(power-1):162+162*(power-1)]
            entry_this = entry[0+162*(power-1):162+162*(power-1)]
            grad_w[0+162*(power-1):162+162*(power-1)] += ( entry_this * temp * (train_this**power) + 2*lamb*w_this ) 
                  

#        for power in range(1,power_num+1):
#            for i in range(162):
#                grad_w[i+162*(power-1)] += entry[i] * temp * (train_this[i]**power)
            
            
        rmse += (ans_this-y)**2 
    
    
    ada_b += grad_b**2
    for i in range(w_num):
        if(entry[i]==0):
            ada_g[i] = 1
        else:
            ada_g[i] += grad_w[i]**2
    
#    ada_b = 1
 #   for i in range(w_num):   
  #      ada_g[i] = 1    

    
    b = b - rate * grad_b / (ada_b**0.5)
    b_table[ite] = b
    for i in range(w_num):
        if(i%18==17):
            w[i] = w[i] - rate * grad_w[i] / (ada_g[i]**0.5)
            w_table[i,ite] = w[i]
        else:
            w[i] = w[i] - rate * grad_w[i] / (ada_g[i]**0.5)
            w_table[i,ite] = w[i]
    
    rmse_table[ite] = ( rmse / sample_num )**0.5

#%%

import numpy as np
import csv

with open('test_X.csv', 'rb') as f_:
    reader_ = csv.reader(f_)
    raw_ = list(reader_)

raw_ = np.array(raw_)
raw_[raw_=='NR']=0


test = []
for i in range(240):
    for j in range(2,11):
        if(i==0 and j==2):
            test = raw_[0:18,2]
        else:
            new = raw_[0+18*i:18+18*i,j]
            test = np.append(test,new)

test = test.astype(np.float) 
            

test_num = 240
pred = np.zeros(test_num)
for n in range(test_num):
    test_this = test[0+162*n:162+162*n]
    
    pred[n] = b
    for power in range(1,power_num+1):
        w_this = w[0+162*(power-1):162+162*(power-1)]
        pred[n] += np.sum(w_this * (test_this**power))        

        
title = np.array([['id','value']])        
ids = raw_[:,0]        
_, idx = np.unique(ids, return_index=True)
id = ids[np.sort(idx)]
         
result = np.c_[id,pred]
result = np.r_[title,result]
np.savetxt("linear_regression.csv", result, delimiter=",", fmt="%s")


