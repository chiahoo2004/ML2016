from __future__ import absolute_import
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import pandas as pd
import sys
import keras
from keras import backend as K
K.set_image_dim_ordering('tf')


import sys
path= sys.argv[1]
output_model = sys.argv[2]


all_label = pickle.load( open(path+'all_label.p','rb') )
all_unlabel = pickle.load( open(path+'all_unlabel.p','rb') )
test = pickle.load( open(path+'test.p','rb') )
all_unlabel = np.array(all_unlabel)

nb_classes = 10




y_train = np.array([]);
for i in range(10):
    all_label_1 = np.reshape(all_label[i], (500,3072))
    if(i==0):
        X_train = all_label_1
    else:
        X_train = np.concatenate((X_train, all_label_1), axis=0)
    labels = i*np.ones(500)
    y_train = np.append(y_train,labels)

#X_train = np.concatenate((X_train, all_unlabel), axis=0)

X_test = np.array( pd.Series(test).values[1] )
y_train = y_train.astype(int)






size=3*32*32

batch_size = 8
nb_classes = 10 
nb_epoch = 5
nb_hidden_layers = [size, 2000, 1000, 500]

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)

# Layer-wise pre-training
trained_encoders = []
X_train_tmp = np.copy(X_train)
X_test_tmp = np.copy(X_test)
for n_in, n_out in zip(nb_hidden_layers[:-1], nb_hidden_layers[1:]):
    print('Pre-training the layer: Input {} -> Output {}'.format(n_in, n_out))
    
    encoding_dim = n_out
    input_img = Input(shape=(n_in,))
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    decoded = Dense(n_in, activation='sigmoid')(encoded)
    Autoencoder = Model(input=input_img, output=decoded)
    
    
    
    Autoencoder.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
    Autoencoder.fit(X_train_tmp, X_train_tmp, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_test_tmp, X_test_tmp))
    ############################################################################################################################################
    
    # Store trainined weight
    trained_encoders.append(Autoencoder.layers[1])
    
    
    temp = Sequential()
    temp.add(Autoencoder.layers[1])
    temp.compile(loss='mean_squared_error', optimizer='rmsprop')
    X_train_tmp = temp.predict(X_train_tmp)
    X_test_tmp = temp.predict(X_test_tmp)












# Fine-tuning
print('Predict')
model = Sequential()
for encoder in trained_encoders:
    model.add(encoder)

model.add(Dense(nb_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch)






X_train_unlabel = all_unlabel.astype("float32") / 255.0
y_train_unlabel = model.predict(X_train_unlabel)

basis = np.identity( nb_classes )
similarity = cosine_similarity(y_train_unlabel, basis)
y_train_unlabel = np.argmax(similarity, axis=1)
Y_train_unlabel = np_utils.to_categorical(y_train_unlabel, nb_classes)

print('y_train_unlabel')
print(np.bincount(y_train_unlabel))







X_train_final = np.concatenate((X_train, X_train_unlabel), axis=0)
Y_train_final = np.concatenate((Y_train, Y_train_unlabel), axis=0)







# Fine-tuning
print('Final')
model = Sequential()
for encoder in trained_encoders:
    model.add(encoder)

model.add(Dense(nb_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit(X_train_final, Y_train_final, batch_size=batch_size, nb_epoch=nb_epoch)		##############################################################################




#model.save(output_model)
#pickle.dump(trained_encoders, open( output_model, 'wb' ), pickle.HIGHEST_PROTOCOL)
model.save_weights(output_model)
