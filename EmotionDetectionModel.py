# Importing the os module to interact with the operating system
import os
# Importing the OpenCV library for image processing
import cv2
# Importing NumPy for numerical operations
import numpy as np
# Importing train_test_split from scikit-learn to split data
from sklearn.model_selection import train_test_split
# Importing Sequential model from Keras for building deep learning models
from keras.models import Sequential
# Importing layers from Keras for model architecture
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# Importing to_categorical from Keras for one-hot encoding of labels
from keras.utils import to_categorical

# Define the paths to folders containing emotion-labeled images : in my case , it locates in Downloads\archive\train
emotionFolders = [
    "C:\\Users\\MSI\\Downloads\\archive\\train\\angry",
    "C:\\Users\\MSI\\Downloads\\archive\\train\\disgust",
    "C:\\Users\\MSI\\Downloads\\archive\\train\\fear",
    "C:\\Users\\MSI\\Downloads\\archive\\train\\happy",
    "C:\\Users\\MSI\\Downloads\\archive\\train\\neutral",
    "C:\\Users\\MSI\\Downloads\\archive\\train\\sad",
    "C:\\Users\\MSI\\Downloads\\archive\\train\\surprise",
]

# Function to load images and corresponding labels from provided folders
def load_images(emotion_folders):
    # List to store images
    images = []
    # List to store corresponding labels
    labels = []
    # Iterate over emotion folders and iterate over files in each folder
    for i, folder in enumerate(emotion_folders):
        for filename in os.listdir(folder):
            # Read image in grayscale
            img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_GRAYSCALE)
            # Resize image to 48x48 pixels
            img = cv2.resize(img, (48,48))
            # Append image to list
            images.append(img)
            # Append label to list
            labels.append(i)
    # Convert lists to numpy arrays
    return np.array(images), np.array(labels)

# Load images and labels from provided folders
images, labels = load_images(emotionFolders)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Reshape and normalize the input images
X_train = X_train.reshape(X_train.shape[0], 48, 48, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1).astype('float32') / 255

# One-hot encode the labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define the neural network model architecture and we'll start by initialize sequential model
model = Sequential()
# Add convolutional layer
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
# Add convolutional layer
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
# Add max pooling layer
model.add(MaxPooling2D(pool_size=(2,2)))
# Add dropout layer
model.add(Dropout(0.25))
# Flatten the output for dense layers
model.add(Flatten())
# Add dense layer
model.add(Dense(128, activation='relu'))
# Add dropout layer
model.add(Dropout(0.5))
# Add output layer with softmax activation
model.add(Dense(7, activation='softmax'))

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, batch_size=64, epochs=30, verbose=1, validation_data=(X_test, y_test))

# Save the trained model
model.save("emotion_detection_model.h5")
