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
from keras import backend as K
K.set_image_dim_ordering('tf')

import sys
path= sys.argv[1]
output_model = sys.argv[2]

batch_size = 32
nb_classes = 10
nb_epoch = 200
data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3



# the data, shuffled and split between train and test sets

all_label = pickle.load( open(path+'all_label.p','rb') )
all_unlabel = pickle.load( open(path+'all_unlabel.p','rb') )
test = pickle.load( open(path+'test.p','rb') )

all_unlabel = np.array(all_unlabel)










y_train = np.array([]);
for i in range(10):
    all_label_1 = np.reshape(all_label[i], (500,3072))
    if(i==0):
        X_train = all_label_1
    else:
        X_train = np.concatenate((X_train, all_label_1), axis=0)
    labels = i*np.ones(500)
    y_train = np.append(y_train,labels)


X_test = np.array( pd.Series(test).values[1] )

y_train = y_train.astype(int)









X_train_reshape = X_train.reshape((5000, 3, 32, 32))
all_unlabel_reshape = all_unlabel.reshape((45000, 3, 32, 32))
X_test_reshape = X_test.reshape((10000, 3, 32, 32))

tmp = np.zeros((5000,32,32,3))
for i in range(5000):
    stack = np.dstack((X_train_reshape[i][1],X_train_reshape[i][2],X_train_reshape[i][0]))
    tmp[i] = np.copy(stack)

X_train_label = np.copy(tmp)

tmp3 = np.zeros((45000,32,32,3))
for i in range(45000):
#    if(i%1000==0):
#        print '#',i
    stack = np.dstack((all_unlabel_reshape[i][1],all_unlabel_reshape[i][2],all_unlabel_reshape[i][0]))
    tmp3[i] = np.copy(stack)

X_train_unlabel = np.copy(tmp3)

tmp2 = np.zeros((10000,32,32,3))
for i in range(10000):
#    if(i%1000==0):
#        print '#',i
    stack = np.dstack((X_test_reshape[i][1],X_test_reshape[i][2],X_test_reshape[i][0]))
    tmp2[i] = np.copy(stack)

X_test = np.copy(tmp2)

X_train_label = X_train_label.astype('float32')
X_train_unlabel = X_train_unlabel.astype('float32')
X_test = X_test.astype('float32')
X_train_label /= 255
X_train_unlabel /= 255
X_test /= 255











print('X_train shape:', X_train_label.shape)
print(X_train_label.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train_label = np_utils.to_categorical(y_train, nb_classes)


model = Sequential()

model.add(Convolution2D(16, 3, 3, border_mode='same',
                        input_shape=X_train_label.shape[1:]))
model.add(Convolution2D(32, 3, 3))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Convolution2D(128, 3, 3))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

'''
model.add(Convolution2D(256, 3, 3, border_mode='same'))
model.add(Convolution2D(512, 3, 3))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
'''
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adamax',
              metrics=['accuracy'])

if not data_augmentation:
    print('Not using data augmentation.')
    history = model.fit(X_train_label, Y_train_label,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
			  validation_split=0.33,
              shuffle=True)

else:
    print('Using real-time data augmentation.')
    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
	featurewise_center=False,  # set input mean to 0 over the dataset
	samplewise_center=False,  # set each sample mean to 0
	featurewise_std_normalization=False,  # divide inputs by std of the dataset
	samplewise_std_normalization=False,  # divide each input by its std
	zca_whitening=False,  # apply ZCA whitening
	rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
	width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
	height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
	horizontal_flip=True,  # randomly flip images
	vertical_flip=False)  # randomly flip images

    datagen.fit(X_train_label)

    history = model.fit_generator(datagen.flow(X_train_label, Y_train_label,
                            batch_size=batch_size),
                            samples_per_epoch=X_train_label.shape[0],
							nb_epoch=nb_epoch)			

'''							
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.figure(0)
plt.plot(history.history['acc'],'b')
plt.plot(history.history['val_acc'],'r')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig('accuracy1.png', dpi=200)
# summarize history for loss
plt.figure(1)
plt.plot(history.history['loss'],'b')
plt.plot(history.history['val_loss'],'r')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig('loss1.png', dpi=200)
'''


count = 0
coverage = False
y_train_label = np.copy(y_train)
X_train_final = np.copy(X_train_label)
y_train_final = np.copy(y_train_label)
Y_train_final = np.copy(Y_train_label)
X_train_unlabel_tmp = np.copy(X_train_unlabel)
	
	
	
while coverage == False:
	
	count += 1
	
	y_train_unlabel = model.predict(X_train_unlabel_tmp)

	probability = np.amax(y_train_unlabel, axis=1)
	
	basis = np.identity( nb_classes )
	similarity = cosine_similarity(y_train_unlabel, basis)
	y_train_unlabel = np.argmax(similarity, axis=1)
	confidence = np.amax(similarity, axis=1)
	
		
	larger = np.where( probability > 0.9 )[0]
	
	print('#{} confident: {}/{}'.format(count, larger.shape[0], y_train_unlabel.shape[0]))
	
	X_train_unlabel_conf = np.copy( X_train_unlabel_tmp[larger] )
	X_train_final = np.concatenate((X_train_final, X_train_unlabel_conf), axis=0)
	X_train_unlabel_tmp = np.delete(X_train_unlabel_tmp, larger, 0)
	Y_train_unlabel_tmp = model.predict(X_train_unlabel_tmp)
	
	y_train_unlabel_conf = np.copy( y_train_unlabel[larger] )
	y_train_final = np.concatenate((y_train_final, y_train_unlabel_conf), axis=0)
	Y_train_final = np_utils.to_categorical(y_train_final, nb_classes)
	
	print('#{} X_train_unlabel: {}/{}'.format(count, X_train_unlabel_tmp.shape[0], X_train_unlabel.shape[0]))
	print('#{} X_train_label: {}/{}'.format(count, X_train_final.shape[0], X_train_label.shape[0]+X_train_unlabel.shape[0]))
	print('#{} y_train_label: {}/{}'.format(count, y_train_final.shape[0], X_train_label.shape[0]+X_train_unlabel.shape[0]))
	
	if( X_train_unlabel_tmp.shape[0] == 0 or count>=5 ):
		coverage = True
	
	coverage = True
	
	print('#{} y_train_final prediction:{}'.format(count, np.bincount(y_train_final)))
	
	
	if not data_augmentation:
	
		history = model.fit(X_train_final, Y_train_final,
				  batch_size=batch_size,
				  nb_epoch=nb_epoch,
				  validation_split=0.33,
				  shuffle=True)
				  
	else:
	
		# this will do preprocessing and realtime data augmentation
		datagen_final = ImageDataGenerator(
			featurewise_center=False,  # set input mean to 0 over the dataset
			samplewise_center=False,  # set each sample mean to 0
			featurewise_std_normalization=False,  # divide inputs by std of the dataset
			samplewise_std_normalization=False,  # divide each input by its std
			zca_whitening=False,  # apply ZCA whitening
			rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
			width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
			height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
			horizontal_flip=True,  # randomly flip images
			vertical_flip=False)  # randomly flip images

		# compute quantities required for featurewise normalization
		# (std, mean, and principal components if ZCA whitening is applied)
		datagen_final.fit(X_train_final)

		# fit the model on the batches generated by datagen.flow()
		history = model.fit_generator(datagen_final.flow(X_train_final, Y_train_final,
							batch_size=batch_size),
							samples_per_epoch=X_train_final.shape[0],
							validation_data=(X_train_unlabel_tmp, Y_train_unlabel_tmp),
							nb_epoch=nb_epoch)			

						

		
'''		
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.figure(2)
plt.plot(history.history['acc'],'b')
plt.plot(history.history['val_acc'],'r')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig('accuracy2.png', dpi=200)
# summarize history for loss
plt.figure(3)
plt.plot(history.history['loss'],'b')
plt.plot(history.history['val_loss'],'r')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig('loss2.png', dpi=200)
'''



model.save(output_model)





	
