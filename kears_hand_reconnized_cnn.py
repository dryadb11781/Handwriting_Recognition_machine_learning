# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 14:12:01 2017

@author: Nami
"""


import numpy as np
np.random.seed(1337)  # for reproducibility
import glob

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

import csv
import numpy as np
files=glob.glob('*.csv')
n = 0
data = []
for temp in files:
    f = open('%s'%temp, 'r') #000.csv 改為自己的手寫辨識資料檔名
    for row in csv.reader(f):
        data.append(row)
    f.close()


#data.remove(data[0])

for n in range(len(data)):
    for i in range(len(data[0])):
        data[n][i] = int(data[n][i])

image = []
target = []

for n in range(len(data)):
    t = data[n][0]
    target.append(t)
    image.append(data[n][1:])

image = np.array(image)
#### up is load data

#####create roate image
image_64_64=image.reshape(400,64,64)
a=np.zeros((1,64,64))
from scipy.ndimage import rotate
for ii in range(1,3):
  for iii in range(image_64_64.shape[0]):
      a[0,:,:] = rotate(image_64_64[iii,:,:], ii*15, reshape=False)
      image_64_64=np.vstack((image_64_64,a))
      target.append(target[iii])

image_4096=np.zeros((image_64_64.shape[0],4096))
for xx in range(image_64_64.shape[0]):
    image_4096[xx,:]=image_64_64[xx,:,:].flatten()

from sklearn.model_selection import train_test_split
image1, image2, target1, target2 = train_test_split(image_64_64, target, test_size = 0.3)
X_train=image1
y_train=target1
X_test=image2
y_test=target2



batch_size = 10
nb_classes = 10
nb_epoch = 20

# input image dimensions
img_rows, img_cols = 64, 64
# number of convolutional filters to use
nb_filters = 16
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets


if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape,name="first_cnn_layer"))
model.add(Activation('relu',name="first_active_layer"))
model.add(MaxPooling2D(pool_size=pool_size,name="first_pooling_layer"))


model.add(Convolution2D(32, kernel_size[0], kernel_size[1],name="second_cnn_layer"))
model.add(Activation('relu',name="second_adctive_layer"))
model.add(MaxPooling2D(pool_size=pool_size,name="second_pooling_layer"))

model.add(Convolution2D(32, kernel_size[0], kernel_size[1],name="three_cnn_layer"))
model.add(Activation('relu',name="three_active_layer"))
model.add(MaxPooling2D(pool_size=pool_size,name="three_pooling_layer"))

model.add(Convolution2D(32, kernel_size[0], kernel_size[1],name="four_cnn_layer"))
model.add(Activation('relu',name="four_active_layer"))
model.add(MaxPooling2D(pool_size=pool_size,name="four_pooling_layer"))


model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(30))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
model.summary()
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test),shuffle=True)
score = model.evaluate(X_test, Y_test, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])
import matplotlib.pyplot as plt
a=model.predict_classes(X_test)
plt.figure()
plt.imshow(X_test[2,:,:,0])
print(a[2])
#
##畫出中間層產物

##test_detect=np.zeros((2,1,64,64))
##test_detect[0,0,16:32:2,:]=1
#plt.matshow(X_train[0,0,:,:], cmap='Greys')
#from keras import backend as K

# with a Sequential model
#get_3rd_layer_output = K.function([model.layers[0].input],
#                                  [model.layers[2].output])
#layer_output = get_3rd_layer_output([X_train])[0]
#
#
#print(layer_output.shape)
#


#
#plt.figure(figsize=[10,10])
#
#for i in range(16):
#    plt.subplot(5,4,1+i)
#    plt.imshow(layer_output[0,i,:,:], cmap='Greys')
#
#plt.show()
#######
#get_3rd_layer_output = K.function([model.layers[0].input],
#                                  [model.layers[11].output])
#layer_output = get_3rd_layer_output([X_train])[0]
#
#
#print(layer_output.shape)
#
#
#plt.figure(figsize=[10,10])
#
#for i in range(16):
#    plt.subplot(5,4,1+i)
#    plt.imshow(layer_output[0,i,:,:], cmap='Greys')
#
#plt.show()

'''


#matshow 是畫單張的
#plt.matshow(layer_output[0,:,:,0], cmap='Greys')
#imshow 是畫subplot的

plt.figure(figsize=[10,10])
#繪圖
for i in range(16):

    plt.subplot(4,4,1+i)
    plt.imshow(layer_output[0,:,:,i], cmap='Greys')

plt.show()


 '''

####下面是存取與讀取模型的作法
'''
print ('我是分隔線')
output_fn = 'cnn_tensorflow_model'


#saving model
from keras.models import load_model
model.save(output_fn + '.h5')
del model

#loading model
model = load_model(output_fn + '.h5')

#prediction
pred = model.predict_classes(X_test, batch_size, verbose=0)
print(pred)
print(pred.shape[0])
ans = [np.argmax(r) for r in Y_test]

# caculate accuracy rate of testing data
acc_rate = sum(pred-ans == 0)/float(pred.shape[0])

print ("Accuracy rate:", acc_rate)



'''
