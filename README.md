# Emotion Detection AI Model

AI model for emotion detection from images using deep learning techniques. The model is capable of identifying seven different emotions: angry, disgust, fear, happy, neutral, sad, and surprise.

## Dataset
The model is trained on a dataset from kaggle containing images labeled with various emotions .
here is the link "https://www.kaggle.com/datasets/msambare/fer2013"

## Installation

Clone this repository to your local machine:

git clone https://github.com/OxFOIQ/EmotionDetectionModel.git

## Install the required dependencies:

Python 3.x

OpenCV

NumPy

scikit-learn

Keras


## Usage
Run the EmotionDetectionModel.py script to train the model:

python EmotionDetectionModel.py

After training, the model will be saved as emotion_detection_model.h5.

You can then use the trained model for emotion detection tasks by loading it in your application.

## Model Architecture

The model architecture consists of convolutional layers followed by max-pooling layers, dropout layers for regularization, and dense layers. The output layer uses softmax activation for multi-class classification.

## Contributing

Contributions to improve the model's accuracy, efficiency, or documentation are welcome. Feel free to fork this repository and submit pull requests.
