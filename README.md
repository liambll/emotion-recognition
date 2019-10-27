# Facial Expression Recognition in Real-Time
- Facial expression classification using Linear Discrimination Analysis (FisherFace) and Convolutional Neural Network
- Demo for real-time Facial expression classification based on image from camera feed
- Technologies: Python, OpenCV, Keras, Amazon AWS

## 1) Requirement:
- Python 3
- OpenCV
- Keras

## 2) Description:
- Data: Facial expression dataset with 500+ images can be downloaded at: http://www.consortium.ri.cmu.edu/ckagree/
- emotion_files.py: pre-process original facial expression images
- emotion_lda: Facial expression classification model training using Linear Discriminant Analysis (LDA) with OpenCV
- emotion_cnn: Facial expression classification model training using Convolutional Neural Network (CNN) with Keras
- camera_lda_model.py: run real-time facial expression classification demo (LDA model) with live feed from webcam
- camera_cnn_model.py: run real-time facial expression classification demo (CNN model) with live feed from webcam


