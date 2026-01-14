import os

#suppressing warning message
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize images (important!)
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define class names
class_names = ["T-shirt/top","Trouser","Pullover","Dress","Coat",
               "Sandal","Shirt","Sneaker","Bag","Ankle boot"]

# Build the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

# Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
model.fit(train_images, train_labels, epochs=20)

# Generate predictions
predictions = model.predict(test_images)

# Visualize predictions for first 5 test images
for i in range(5):
    plt.imshow(test_images[i], cmap='gray')
    plt.title(f"Predicted: {class_names[predictions[i].argmax()]}, True: {class_names[test_labels[i]]}")
    plt.axis('off')
    plt.show()
