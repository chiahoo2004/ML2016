import pandas as pd
import os
import re
from nltk.corpus import stopwords
import nltk.data
import logging
import numpy as np  # Make sure that numpy is imported
from gensim.models import word2vec
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from itertools import chain
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

import sys
path= sys.argv[1]
output_file = sys.argv[2]
	
docs = []
with open(path+'docs.txt','rb') as file:
    for line in file:
        if not line.isspace():
            docs.append(line)
                       
docs = ''.join(docs)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

raw_sentences = tokenizer.tokenize(docs.decode('utf-8'))
sentences = []
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
		text = raw_sentence		  
		text = re.sub("[^a-zA-Z]"," ", text)
		words = text.lower().split()
		stops = set(stopwords.words("english"))
		words = [w for w in words if not w in stops]
		sentences.append(words)
		  
# np.save('sentences',sentences)		  

model = word2vec.Word2Vec(sentences, workers=4, \
            size=300, min_count = 40, \
            window = 10, sample = 1e-3)

model_name = "model"
model.save(model_name)		  


from sklearn.cluster import KMeans
word_vectors = model.syn0

'''
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components = 100)
word_vectors = svd.fit_transform(word_vectors)
'''

kmeans_clustering = KMeans( n_clusters = word_vectors.shape[0] / 5 )
idx = kmeans_clustering.fit_predict( word_vectors )
	 

	 
	 
import pandas as pd
import os
import re
from nltk.corpus import stopwords
import nltk.data
import logging
import numpy as np
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier

title = []
with open(path+'title_StackOverflow.txt','rb') as file:
    for line in file:
        if not line.isspace():
            title.append(line)

titlegg = []
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
for sentence in title:
    sentence = re.sub("[^a-zA-Z]"," ", sentence)
    words = sentence.lower().split()
    temp = [w for w in words if not w in stop]
    titlegg.append(temp)
			
test_sentences_list = []
for raw_sentence in title:
    if len(raw_sentence) > 0:
		text = raw_sentence		  
		text = re.sub("[^a-zA-Z]"," ", text)
		words = text.lower().split()
		stops = set(stopwords.words("english"))
		words = [w for w in words if not w in stops]
		test_sentences_list.append(words)		  
		  
		  
# np.save('test_sentences_list',test_sentences_list)	  
# model = Word2Vec.load('model')		  

word_vectors = model.syn0
num_feature = word_vectors.shape[1] 
		  
known = 0
unknown = 0	  
pred = []
test_sentences_mat = np.zeros([ len(test_sentences_list) , num_feature] )
ite = 0
for test_sentences in test_sentences_list:
    test_word_vec = []
    for test_word in test_sentences:
        if test_word in model.index2word:
            test_word_vec.append(model[test_word])
            known += 1
        else:
            unknown += 1
            continue
    test_word_vec = np.array(test_word_vec)
    if(test_word_vec.size==0):
        ite += 1
    else:
        test_sentences_vec = np.mean(test_word_vec,axis=0)
        test_sentences_mat[ite,:] = test_sentences_vec
        ite += 1

thre = 20
titlegg_1d = list(chain.from_iterable(titlegg))
counter = Counter(titlegg_1d)

stats = counter.most_common( thre )
guess = [x[0] for x in stats]

ite = 0
guess_list = [[] for x in range(20000)]
for piece in titlegg:
    guess_list[ite] = set(piece).intersection( set(guess) )
    ite += 1
		
		
pred = kmeans_clustering.predict( test_sentences_mat )
unique, counts = np.unique(pred, return_counts=True)
stats = dict(zip(unique, counts))


import csv

with open(path+'check_index.csv', 'rb') as f:
    reader = csv.reader(f)
    check = list(reader)
check = np.array(check)

check = np.delete(check, 0, 0)
ids = check[:,0]
check = np.delete(check, 0, 1)
x_ID = check[:,0]
y_ID = check[:,1] 
x_ID = x_ID.astype(np.int)
y_ID = y_ID.astype(np.int)

x_pred = pred[x_ID]
y_pred = pred[y_ID]

equal = (x_pred == y_pred)
equal[equal==True] = 1
equal[equal==False] = 0


equal = np.zeros(check.shape[0],)
for i in range(check.shape[0]): 
    match = set( guess_list[x_ID[i]] ).intersection(guess_list[y_ID[i]])
    if( len(match)==0 ):
        equal[i] = 0
    else:
        equal[i] = 1


header = np.array([['ID','Ans']])                
ids = ids.astype(np.int)
equal = equal.astype(np.int)       
result = np.c_[ids,equal]
result = np.r_[header,result]
np.savetxt(output_file, result, delimiter=",", fmt="%s")








	  
