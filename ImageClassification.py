import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import glob as gb
import cv2
import tensorflow as tf
import keras
from keras import layers, optimizers, models
from tensorflow.keras.applications import ResNet50

"""
Define the paths to the train test and pred folders
"""
trainpath = '/content/seg_train/'
testpath = '/content/seg_test/'
predpath = '/content/seg_pred/'


"""
Integer encoding - Dictionary with 6 categories & function to retreive the encoding
"""
code_to_num = {'buildings': 0, 'forest': 1, 'glacier': 2, 'mountain': 3, 'sea': 4, 'street': 5}
num_to_code = {0: 'buildings', 1: 'forest', 2: 'glacier', 3: 'mountain', 4: 'sea', 5: 'street'}

def get_code(n):
    if n in num_to_code:
        return num_to_code[n]

def get_num(c):
    if c in code_to_num:
        return code_to_num[c]


"""
Reading & Resizing of test, train & pred images without loss of accuracy
"""
# size of image
s = 100

X_train = []
y_train = []
for folder in os.listdir(trainpath + 'seg_train'):
    files = trainpath + 'seg_train//' + folder + '/*.jpg'
    for file in files:
        image = cv2.imread(file)
        image_array = cv2.resize(image, (s, s))
        X_train.append(list(image_array))
        y_train.append(get_num(folder))

X_test = []
y_test = []
for folder in os.listdir(testpath + 'seg_test'):
    files = testpath + 'seg_test//' + folder + '/*.jpg'
    for file in files:
        image = cv2.imread(file)
        image_array = cv2.resize(image, (s, s))
        X_test.append(list(image_array))
        y_test.append(get_num(folder))

X_pred = []
files = predpath + 'seg_pred/*.jpg'
for file in files:
    image = cv2.imread(file)
    image_array = cv2.resize(image, (s, s))
    X_pred.append(list(image_array))

#Converting into array
X_train = np.array(X_train)
X_test = np.array(X_test)
X_pred_array = np.array(X_pred)
y_train = np.array(y_train)
y_test = np.array(y_test)


"""
Visualsation of required Prediction Images
"""
plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(X_pred),36))) :
    plt.subplot(6,6,n+1)
    plt.imshow(X_pred[i])
    plt.axis('off')
    

"""
CNN Architecture for image classification 
"""
KerasModel = models.Sequential()
KerasModel.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
KerasModel.add(layers.MaxPooling2D((2, 2)))
KerasModel.add(layers.Conv2D(64, (3, 3), activation='relu'))
KerasModel.add(layers.MaxPooling2D((2, 2)))
KerasModel.add(layers.Conv2D(128, (3, 3), activation='relu'))
KerasModel.add(layers.MaxPooling2D((2, 2)))
KerasModel.add(layers.Conv2D(128, (3, 3), activation='relu'))
KerasModel.add(layers.MaxPooling2D((2, 2)))
KerasModel.add(layers.Flatten())
KerasModel.add(layers.Dropout(0.5))
KerasModel.add(layers.Dense(512, activation='relu'))
KerasModel.add(layers.Dense(6, activation='softmax'))
KerasModel.compile(optimizer=optimizers.Adam(lr=0.001), loss=['sparse_categorical_crossentropy'], metrics=['accuracy'])

#Model Summary
print('Model Details are : ')
print(KerasModel.summary())


"""
Training of model & Evaluating Test Loss & Accuracy
"""
ThisModel = KerasModel.fit(X_train, y_train, epochs=50, batch_size=256, validation_data=(X_test, y_test))
ModelLoss, ModelAccuracy = KerasModel.evaluate(X_test, y_test)
print('Test Loss is {}'.format(ModelLoss))
print('Test Accuracy is {}'.format(ModelAccuracy))


"""
Obtaining Prediction for the unlabeled data
"""
y_pred = KerasModel.predict(X_test)
y_result = KerasModel.predict(X_pred_array)


"""
Random predicted pictures & its predicting category
"""
plt.figure(figsize=(20, 20))
for n, i in enumerate(list(np.random.randint(0, len(X_pred), 36))):
    plt.subplot(6, 6, n + 1)
    plt.imshow(X_pred_array[i])
    plt.axis('off')
    plt.title(get_code(np.argmax(y_result[i])))
    
 

# Transfer Learning Approach - Using ResNet50
conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=(100, 100, 3))

datagen = ImageDataGenerator(
    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)\n",
    zoom_range=0.1,  # Randomly zoom image\n",
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)\n",
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)\n",
    horizontal_flip=True,  # randomly flip images horizontally\n",
    vertical_flip=False,  # Don't randomly flip images vertically\n",
)

batch_size = 32
img_iter = datagen.flow(X_train, y_train, batch_size=batch_size)

# Making last layer trainable and all other layers are static.
for layer in conv_base.layers[:143]:
    layer.trainable = False

conv_base.trainable = True


"""
Transfer Learning Model
"""
KerasModel2 = models.Sequential()
KerasModel2.add(conv_base)
KerasModel2.add(layers.Flatten())
KerasModel2.add(layers.Dropout(0.5))
KerasModel2.add(layers.Dense(64, activation='relu'))
KerasModel2.add(layers.Dense(6, activation='softmax'))

KerasModel2.compile(optimizer=optimizers.Adam(learning_rate=0.00001), loss=['sparse_categorical_crossentropy'],
                    metrics=['accuracy'])


"""
Training of model & Evaluating Test Loss & Accuracy
"""
ThisModel = KerasModel2.fit(img_iter, epochs=3, steps_per_epoch=len(X_train) / batch_size)
ModelLoss, ModelAccuracy = KerasModel2.evaluate(X_test, y_test)
print('Test Loss is {}'.format(ModelLoss))
print('Test Accuracy is {}'.format(ModelAccuracy))
