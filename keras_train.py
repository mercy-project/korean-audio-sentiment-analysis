import numpy as np
import os
import sys
import librosa
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import TimeDistributed
from keras.models import Sequential
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, Activation,Dropout, Flatten, Embedding, GlobalAveragePooling2D
import pickle
from xgboost import XGBClassifier

#Path to the folder consisting different Emotions folders
# path_data = sys.argv[1]
#
# mslen = 22050
#
# data = []
#
# max_fs = 0
# labels = []
#
# emotions = ['neutral','calm','happy','sad','angry','fearful','disgust','surprised']
# directories = os.listdir(path_data)
#
# print(directories)

'''
f2 = open('feature.pkl','rb')
feature_all = pickle.load(f2)

f3 = open('label.pkl','rb')
labels = pickle.load(f3)

from copy import deepcopy
y = deepcopy(labels)
for i in range(len(y)):
    y[i] = int(y[i])



n_labels = len(y)
n_unique_labels = len(np.unique(y))
one_hot_encode = np.zeros((n_labels,n_unique_labels))
f = np.arange(n_labels)
for i in range(len(f)):
    one_hot_encode[f[i],y[i]-1]=1
'''

#TODO
IMG_WIDTH = 256
IMG_HEIGHT = 256
N_LABELS = 7
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True, validation_split = 0.3)

train_generator = train_datagen.flow_from_directory(
    directory=r"./data/train/",
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    subset = 'training'
)

validation_generator = train_datagen.flow_from_directory(
    directory=r"./data/train/",
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    subset = 'validation'
)

#
# #X_train, X_test, y_train, y_test = train_test_split(feature_all, one_hot_encode, test_size = 0.3, random_state=20)
#
# ########################### MODEL 1 ###########################
# model = Sequential()
#
# model.add(Dense(IMG_WIDTH,input_dim =IMG_HEIGHT,init='normal',activation ='relu'))
#
# model.add(Dense(400,init='normal',activation ='relu'))
#
# model.add(Dropout(0.2))
#
# model.add(Dense(200,init='normal',activation ='relu'))
#
# model.add(Dropout(0.2))
#
# model.add(Dense(100,init='normal',activation ='relu'))
#
# model.add(Dropout(0.2))
#
# model.add(Dense(N_LABELS,init='normal',activation ='softmax'))
#
# model.compile(loss = 'categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
#
# #model.fit(X_train,y_train,nb_epoch=200,batch_size = 5,verbose=1)
# model.fit_generator(train_generator,
# steps_per_epoch=15,
# epochs=50,
# validation_data=validation_generator,
# validation_steps=5)
#
# #model.evaluate(X_test,y_test)
# model.evaluate_generator(validation_generator,steps=5)
#
# mlp_model = model.to_json()
# with open('mlp_model_relu_adadelta.json','w') as j:
#     j.write(mlp_model)
# model.save_weights("mlp_relu_adadelta_model.h5")
#
# y_pred_model1 = model.predict_generator(validation_generator,steps=5)
# y2 = np.argmax(y_pred_model1,axis=1)
# y_test2 = np.argmax(y_test , axis = 1)
#
# count = 0
# for i in range(y2.shape[0]):
#     if y2[i] == y_test2[i]:
#         count+=1
#
# print('Accuracy for model 1 : ' + str((count / y2.shape[0]) * 100))
#
# ########################### MODEL 2 ###########################
# model2 = Sequential()
#
# model2.add(Dense(X_train.shape[1],input_dim =X_train.shape[1],init='normal',activation ='relu'))
#
# model2.add(Dense(400,init='normal',activation ='tanh'))
#
# model2.add(Dropout(0.2))
#
# model2.add(Dense(200,init='normal',activation ='tanh'))
#
# model2.add(Dropout(0.2))
#
# model2.add(Dense(100,init='normal',activation ='sigmoid'))
#
# model2.add(Dropout(0.2))
#
# model2.add(Dense(y_train.shape[1],init='normal',activation ='softmax'))
#
# model2.compile(loss = 'categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
#
# #model2.fit(X_train, y_train, nb_epoch=200, batch_size = 5, verbose=1)
# model2.fit_generator(train_generator,
#                     steps_per_epoch=15,
#                     epochs=50,
#                     validation_data=validation_generator,
#                     validation_steps=5)
#
# model2.evaluate_generator(test_generator,steps=5)
#
#
# mlp_model2 = model2.to_json()
# with open('mlp_model_tanh_adadelta.json','w') as j:
#     j.write(mlp_model2)
# model2.save_weights("mlp_tanh_adadelta_model.h5")
#
# y_pred_model2 = model2.predict(X_test)
# y22 = np.argmax(y_pred_model2,axis=1)
# y_test22 = np.argmax(y_test , axis = 1)
#
# count = 0
# for i in range(y22.shape[0]):
#     if y22[i] == y_test22[i]:
#         count+=1
#
# print('Accuracy for model 2 : ' + str((count / y22.shape[0]) * 100))
#
# X_train2,X_test2,y_train2,y_test2 = train_test_split(feature_all, y, test_size = 0.3, random_state=20)

########################### MODEL 3 ###########################
model3 = XGBClassifier()

#TODO fit generator
model3.fit_generator(train_generator,
                     steps_per_epoch=15,
                     epochs=50,
                     validation_data=validation_generator,
                     validation_steps=5)
model3.evals_result()

#TODO
score = cross_val_score(model3, X_train2, y_train2, cv=5)

#TODO predict generator
y_pred3 = model3.predict_generator(validation_generator,steps=5)

count = 0
for i in range(y_pred3.shape[0]):
    # if y_pred3[i] == y_test2[i]:
    #     count+=1
    #TODO compare with actual label
    if y_pred3[i] == validation_generator.next():
        count+=1

print('Accuracy for model 3 : ' + str((count / y_pred3.shape[0]) * 100))

# ########################### MODEL 4 ###########################
# img_width = 256
# img_height = 256
# img_channel = 3
# model4 = InceptionV3(include_top=False, weights='imagenet', input_shape=(img_width,img_height,img_channel), pooling=None, classes=7)
# last = model4.output
# x = GlobalAveragePooling2D()(last)
# x = Dense(512, activation='relu')(x)
# preds = Dense(7, activation='softmax')(x)
# for layer in model4.layers:
#     layer.trainable = False



########################### TESTING ###########################
test_file_path = sys.argv[2]
X,sr = librosa.load(test_file_path, sr = None)
stft = np.abs(librosa.stft(X))

############# EXTRACTING AUDIO FEATURES #############
mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40),axis=1)

chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)

mel = np.mean(librosa.feature.melspectrogram(X, sr=sr).T,axis=0)

contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr,fmin=0.5*sr* 2**(-6)).T,axis=0)

tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sr*2).T,axis=0)

features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])

feature_all = np.vstack([feature_all,features])


x_chunk = np.array(features)
x_chunk = x_chunk.reshape(1,np.shape(x_chunk)[0])
for i in [model3] :
    #TODO predict generator
    y_chunk_model1 = i.predict(x_chunk)
    index = np.argmax(y_chunk_model1)
    print('Emotion index',index)
