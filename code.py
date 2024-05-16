import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam, RMSprop, SGD

# Set the paths
data_dir = r'D:\School\machinelearn\midterm\data\train'
test_dir = r'D:\School\machinelearn\midterm\data\test'

# Constants
img_size = 224
batch_size = 128
num_classes = 10
epochs = 50

# Data augmentation and preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    validation_split=0.2
)
test_datagen=ImageDataGenerator(
    rescale=1./255
)

# Load the data
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    class_mode='categorical',
    batch_size=batch_size,
    subset='training',
    shuffle=True
)
val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    class_mode='categorical',
    batch_size=batch_size,
    subset='validation',
    shuffle=True
)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_size, img_size),
    class_mode='categorical'
)

car_names=list(train_generator.class_indices.keys())

# Define the sequential model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(img_size, img_size, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
print(model.summary())

# Compile the model
model.compile(optimizer=Adam(), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=epochs,
    batch_size=batch_size,
    verbose=1,
    validation_data=val_generator
)

# Output training loss and accuracy
train_loss, train_acc = model.evaluate(train_generator)
print(f'Training accuracy: {train_acc}, Training loss: {train_loss}')

# Evaluate the model
val_loss, val_acc = model.evaluate(val_generator)
print(f'Validation accuracy: {val_acc}, Validation loss: {val_loss}')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(epochs), acc, label='Training Accuracy')
plt.plot(range(epochs), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(epochs), loss, label='Training Loss')
plt.plot(range(epochs), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Load a batch of images from the test dataset
batch_images, batch_labels = next(test_generator)

# Make predictions using the trained model
predictions = model.predict(batch_images)

# Plot the images along with their predicted labels
plt.figure(figsize=(15, 7))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(batch_images[i])
    predicted_label = np.argmax(predictions[i])
    true_label = np.argmax(batch_labels[i])
    confidence_score = round(predictions[i][predicted_label]*100, 2)
    plt.title(f'Predicted: {car_names[predicted_label]},\nActual: {car_names[true_label]}.\nConfidence: ({confidence_score:}%)')
    plt.axis('off')
plt.show()