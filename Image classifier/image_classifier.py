#Import Necessary Libraries
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers,datasets,models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

#Load MNIST dataset
(train_images,train_labels),(test_images,test_labels) = datasets.mnist.load_data()

#Preprocessing : Normalise the values between 0 and 1
train_images = train_images/255.0
test_images = test_images/255.0

#Reshape the images to (28,28,1) as they are greyscale
train_images = train_images.reshape((train_images.shape[0],28,28,1))
test_images = test_images.reshape((test_images.shape[0],28,28,1))

#Convert the labels to one-hot encoded format
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#Build the CNN Model
model = models.Sequential()

#First Convolutional Layer
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
#Second Convolutional Layer
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
#Third Convolutional Layer
model.add(layers.Conv2D(64,(3,3),activation='relu'))

#Flatten the 3D output to 1D and add a Dense layer
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))

#Output layer with 10 neurons (for 10 digit classifier)
model.add(layers.Dense(10,activation='softmax'))

#Compile the model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=["accuracy"])

#train the model
model.fit(train_images,train_labels,epochs=5,batch_size=64,validation_data=(test_images,test_labels))

#Evaluate the model on test data
test_loss,test_acc = model.evaluate(test_images,test_labels)
print(f"Test accuracy: {test_acc}")
print(f"Test loss : {test_loss}")

#Make prediction on test images
prediction = model.predict(test_images)
print(f"Prediction of the 12th image : {np.argmax(prediction[12])}")

plt.imshow(test_images[12].reshape(28,28),cmap="gray")
plt.title(f"predicted label : {np.argmax(prediction[12])}")
plt.show()

model.save("img_classifier.keras")