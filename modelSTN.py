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

class SpatialTransformer(layers.Layer):
    def __init__(self, output_size):
        super(SpatialTransformer, self).__init__()
        self.output_size = output_size

    def build(self, input_shape):
        self.num_channels = input_shape[3]

        # Initialize the transformation parameters to identity matrix
        self.theta = self.add_weight(shape=(input_shape[0], 6),
                                      initializer='zeros',
                                      trainable=True)
        
    def _bilinear_sampler(self, inputs):
        images, theta = inputs
        batch_size, height, width, num_channels = tf.shape(images)[0], tf.shape(images)[1], tf.shape(images)[2], tf.shape(images)[3]

        # Generate grid
        grid = self._meshgrid(height, width)

        # Reshape grid to [batch_size, 2, H*W]
        grid = tf.reshape(grid, [1, 2, -1])

        # Transform grid using affine transform
        T_g = tf.linalg.matmul(theta, grid)
        x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
        y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])

        # Reshape to [batch_size, H, W, 2]
        x_s = tf.reshape(x_s, [batch_size, height, width, 1])
        y_s = tf.reshape(y_s, [batch_size, height, width, 1])

        # Sample pixels using bilinear interpolation
        return self._bilinear_sampler_sample(images, x_s, y_s)

    def _bilinear_sampler_sample(self, images, x, y):
        batch_size, height, width, num_channels = tf.shape(images)[0], tf.shape(images)[1], tf.shape(images)[2], tf.shape(images)[3]

        x = tf.cast(x, dtype=tf.float32)
        y = tf.cast(y, dtype=tf.float32)
        height_f = tf.cast(height, dtype=tf.float32)
        width_f = tf.cast(width, dtype=tf.float32)

        # Scale indices from [-1, 1] to [0, width/height - 1]
        x = 0.5 * ((x + 1.0) * width_f - 1.0)
        y = 0.5 * ((y + 1.0) * height_f - 1.0)

        # Compute pixel positions and indices for each channel
        x0 = tf.cast(tf.floor(x), dtype=tf.int32)
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), dtype=tf.int32)
        y1 = y0 + 1

        # Clamp indices to image boundary
        x0 = tf.clip_by_value(x0, 0, width - 1)
        x1 = tf.clip_by_value(x1, 0, width - 1)
        y0 = tf.clip_by_value(y0, 0, height - 1)
        y1 = tf.clip_by_value(y1, 0, height - 1)

        # Compute indices for four corners
        indices00 = tf.reshape(tf.range(batch_size) * height * width + y0 * width + x0, [-1])
        indices01 = tf.reshape(tf.range(batch_size) * height * width + y1 * width + x0, [-1])
        indices10 = tf.reshape(tf.range(batch_size) * height * width + y0 * width + x1, [-1])
        indices11 = tf.reshape(tf.range(batch_size) * height * width + y1 * width + x1, [-1])

        # Gather pixel values at four corners
        pixel_values_00 = tf.gather(tf.reshape(images, [-1, num_channels]), indices00)
        pixel_values_01 = tf.gather(tf.reshape(images, [-1, num_channels]), indices01)
        pixel_values_10 = tf.gather(tf.reshape(images, [-1, num_channels]), indices10)
        pixel_values_11 = tf.gather(tf.reshape(images, [-1, num_channels]), indices11)

        # Compute bilinear weights
        wa = tf.expand_dims((x1 - x) * (y1 - y), axis=-1)
        wb = tf.expand_dims((x1 - x) * (y - y0), axis=-1)
        wc = tf.expand_dims((x - x0) * (y1 - y), axis=-1)
        wd = tf.expand_dims((x - x0) * (y - y0), axis=-1)

        # Compute output
        output = tf.add_n([wa * pixel_values_00, wb * pixel_values_01, wc * pixel_values_10, wd * pixel_values_11])
        return output

    def _meshgrid(self, height, width):
        x_t, y_t = tf.meshgrid(tf.range(width), tf.range(height))
        x_t = tf.cast(x_t, dtype=tf.float32)
        y_t = tf.cast(y_t, dtype=tf.float32)
        ones = tf.ones_like(x_t)
        grid = tf.stack([x_t, y_t, ones], axis=-1)
        return grid

    def call(self, inputs):
        # Split the input tensor into images and theta
        images = inputs[:, :, :, :3]
        theta = inputs[:, :, :, 3:]

        # Perform spatial transformation
        return self.stn([images, self.localization(images)])


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

# Define the CNN architecture with STN layer
model = models.Sequential([
    SpatialTransformer(output_size=(64, 64)),  # STN layer
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(512, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(np.max(labels) + 1, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Learning rate scheduler callback
lr_callback = LearningRateScheduler(lr_scheduler)

# Train the model using augmented data for training and validation data for validation
history = model.fit(train_augmented_data_gen, epochs=10, validation_data=valid_data_gen, callbacks=[lr_callback])

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
model.save("trained_cnn_model_with_stn.h5")