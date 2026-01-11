Task-04: Hand Gesture Recognition
Objective
To develop a hand gesture recognition model that classifies different hand gestures from image data, enabling gesture-based human–computer interaction.

Dataset
Source: Kaggle – Leap Gesture Recognition

Link: https://www.kaggle.com/datasets/gti-upm/leapgestrecog
Dataset contains multiple hand gesture classes organized in nested folders.

Approach
Load gesture images from nested folders
Resize images and convert to grayscale
Flatten image pixels into feature vectors
Normalize pixel values
Train an SVM (Support Vector Machine) classifier
Evaluate classification accuracy

Training Note
The original dataset contains a large number of images.
For local training and testing, a subset of images per gesture was used to reduce training time and memory usage.
For GitHub submission, only a few sample images are uploaded for reference, while the full dataset was used during experimentation.

Technologies Used
Python
OpenCV (cv2)
NumPy
scikit-learn

Project Structure
prodigy/
│── task4.py
│── leapGestRecog/
│── README.md
│── .venv/

Conclusion

The model successfully classifies multiple hand gestures using classical machine learning techniques, providing a lightweight and interpretable solution for gesture recognition tasks.
