# Author: LB
# Read input from Kaggle data and save it to 7 folders corresponding to 7 emotions
# You need to create these 7 folders first: "anger", "disgust", "fear", "happy", "sad", "surprise","neutral"
import cv2
import os
import numpy as np
import pandas as pd

#(0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
emotions = ["anger", "disgust", "fear", "happy", "sad", "surprise","neutral"]

inputs = 'D:/Training/Source_Code/Python/MachineLearning/Project'
df_train = pd.read_csv(inputs+'/fer2013/fer2013.csv')

#Quick Sumamry
df_train.shape[0]
df_train.count
df_train.groupby('emotion').count()

for i in np.arange(df_train.shape[0]):
    pixels = df_train.iloc[i,1]
    pixels_list = pixels.split()
    pixels_matrix = np.asarray(pixels_list).reshape(48,48).astype(np.uint8)
    #len(pixels_matrix)
    label = emotions[df_train.iloc[i,0]]
    fullPath = inputs+"/fer2013_classified/"+label
    fileNumber = len(os.listdir(fullPath))
    cv2.imwrite(fullPath+"/img_"+str(fileNumber)+".png",pixels_matrix)

#show image
#cv2.imshow('my webcam', pixels_matrix)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
