# Importing OpenCV library for computer vision tasks
import cv2
# Importing load_model function from Keras to load pre-trained model
from keras.models import load_model
# Importing NumPy for numerical operations
import numpy as np

# Load Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load pre-trained emotion detection model
emotion_model = load_model("emotion_detection_model.h5")

# Define the emotions labels
EMOTIONS = ["Angry", "Digust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Function to detect emotion from a face image
def detect_emotion(face_image):
    # Resize the face image to match the input size of the model
    face_image = cv2.resize(face_image, (48, 48))
    # Convert face image to grayscale
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    # Reshape the image for model prediction
    face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])
    # Predict the emotion class using the loaded model
    predicted_class = np.argmax(emotion_model.predict(face_image))
    # Return the predicted emotion label
    return EMOTIONS[predicted_class]

# Open the default camera
cap = cv2.VideoCapture(0)

# Main loop for real-time emotion detection
while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # Iterate through each detected face
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Extract the face region from the frame
        face_image = frame[y:y + h, x:x + w]
        # Detect emotion from the face region
        emotion = detect_emotion(face_image)
        # Put text of detected emotion on the frame
        cv2.putText(frame, emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame with emotion detection
    cv2.imshow("Emotion Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
