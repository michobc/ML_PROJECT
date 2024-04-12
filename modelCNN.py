import numpy as np
import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler

# Load and preprocess the data
image_folder_path = "./data"
data = []
labels = []

for folder in os.listdir(image_folder_path):
    subfolder_path = os.path.join(image_folder_path, folder)
    if os.path.isdir(subfolder_path):
        for filename in os.listdir(subfolder_path):
            image_path = os.path.join(subfolder_path, filename)
            if os.path.isfile(image_path) and image_path.endswith(('.jpg', '.jpeg', '.png', '.ppm')):
                image = cv2.imread(image_path)  # Load color image
                if image is not None and not image.size == 0:
                    resized_image = cv2.resize(image, (64, 64))
                    data.append(resized_image)
                    labels.append(folder)

data = np.array(data)
labels = np.array(labels)

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Normalize the pixel values to the range [0, 1]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Define augmentation parameters for training data
train_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')

# Fit the generator to your training data
train_datagen.fit(x_train)

# Create augmented training data batches
train_augmented_data_gen = train_datagen.flow(x_train, y_train, batch_size=64)

# Define validation data generator without augmentation
valid_datagen = ImageDataGenerator()

# Create validation data batches
valid_data_gen = valid_datagen.flow(x_test, y_test, batch_size=64)

# Define learning rate scheduler
def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# Define the CNN architecture
model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),  # Increase number of filters
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(512, (3, 3), activation='relu'),  # Increase number of filters
    layers.Flatten(),
    layers.Dense(256, activation='relu'),  # Reduce dropout and dense layer units
    layers.Dense(np.max(labels) + 1, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Learning rate scheduler callback
lr_callback = LearningRateScheduler(lr_scheduler)

# Train the model using augmented data for training and validation data for validation
history = model.fit(train_augmented_data_gen, epochs=5, validation_data=valid_data_gen, callbacks=[lr_callback])

# Visualize some sample predictions
predictions = model.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)

# Decode labels
decoded_labels = label_encoder.inverse_transform(predicted_classes)

# Plot some sample predictions
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[i])  # Display color image
    plt.title("Predicted: " + decoded_labels[i])
    plt.axis('off')
plt.show()

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_acc)

# Save the trained model
model.save("trained_cnn_model_updated.h5")