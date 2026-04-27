# Hand Gesture Recognition

## Description

This project is part of my Machine Learning internship task 4. It implements a real-time hand gesture recognition system using computer vision and a K-Nearest Neighbors (KNN) model to classify hand gestures from webcam input.

## Dataset

No external dataset is used.
Currently, the model is trained on **sample/dummy data**.
Hand landmarks (21 points → 63 features) are extracted using MediaPipe.

## Tools Used

Python, OpenCV, MediaPipe, NumPy, Scikit-learn

## Method

K-Nearest Neighbors (KNN) with hand landmark feature extraction

## Steps Followed

* Captured live video using webcam
* Detected hand landmarks using MediaPipe
* Converted landmarks into feature vectors
* Trained KNN model
* Predicted gestures in real-time

## Result

The system successfully detects hand landmarks and displays predicted gesture labels in real-time.

## Author

Purva Wagh
