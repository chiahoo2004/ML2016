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

tree = [ Node(i) for i in range(4001) ] ###################################
size = 0
    
def build(L, R, n, train):
    
    global tree
    global size
    feature_num = 57    ###################################
    class_type = 2      ###################################
    
    total = np.zeros(class_type)
    for i in range(class_type):
        total[i] = np.sum( train[L:R+1,feature_num]==i )
    
    ss = R-L+1
    for c in range(class_type):
        if( total[c]==ss ):
            tree[n].result = c;
            return 
    
    minimum = 1e9
    for d in range(feature_num):
        
        dim = d
        
        temp = train[L:R+1]
        temp = temp[ temp[:,dim].argsort() ]
        train[L:R+1] = temp

        numL = 0
        count = np.zeros(class_type)
        for i in range(L,R):
            
            count[ train[i][feature_num] ] += 1
            numL += 1
            
            if(train[i][d] == train[i+1][d]):
                continue
            
            numR = ss - numL
            giniL = numL**2
            for c in range(class_type):
                giniL -= count[c]**2
            giniR = numR**2
            for c in range(class_type):
                giniR -= (total[c]-count[c])**2

            gini = float(giniL)/numL + float(giniR)/numR
            
            if(gini<minimum):
                minimum = gini
                M = i
                dimension = d
                threshold = train[i][d]
            
            dummy = 0

    if(minimum==1e9):
        tree[n].result = 1e9
        return
        
    size += 1
    left = size
    size += 1
    right = size
    
    
    tree[n].left = left
    tree[n].right = right
    tree[n].dimension = dimension
    tree[n].threshold = threshold
    tree[n].result = -1; 
    dim = dimension
    
    temp = train[L:R+1]
    temp = temp[ temp[:,dim].argsort() ]
    train[L:R+1] = temp
    
    build(L,M,left,train)
    build(M+1,R,right,train)              

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

def decision_tree(X_train, y_train):
    
    train = np.concatenate((X_train, y_train), axis=1)
    
    N = y_train.shape[0]
#    N = 4001        ###################################
    build(0, N-1, 0, train)

if __name__ == '__main__':
    
    global tree
    
    
    training_data = sys.argv[1]
    output_model = sys.argv[2]
    
    
    with open(training_data, 'rb') as f:
        reader = csv.reader(f)
        raw = list(reader)

    raw = np.array(raw)
    raw = raw.astype(np.float) 
    y_train = raw[:,-1:]
    X_train = np.delete(raw, 0, 1)
    X_train = np.delete(X_train, -1, 1)
    
    
    
    decision_tree(X_train, y_train)
    acc_train = validate(X_train, y_train)
    print('Acc_train: %.10f' % (acc_train))
    
    '''
    for i in range(4001):
        print('\ntree[i].left:',tree[i].left)
        print('\ntree[i].right:',tree[i].right)
        print('\ntree[i].dimension:',tree[i].dimension)
        print('\ntree[i].threshold:',tree[i].threshold)
        print('\ntree[i].result:',tree[i].result)
    '''
    
    tree_np = np.array(tree)
    np.save(output_model, tree_np)
#    pickle.dump(tree_np, open( "model.pkl", "wb" ), pickle.HIGHEST_PROTOCOL)
    
    