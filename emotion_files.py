# Author: LB
# Copy original images to 8 folders, each corresponding to an emotion

import glob
from shutil import copyfile
import cv2

# Get emotions from txt files and copy corresponding images
emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Define emotion order
participants = glob.glob("D:\\Training\\Source_Code\\Python\\MachineLearning\\Project\\Emotion\\*") #Returns a list of all folders with participant numbers

for x in participants:
    #x = participants[0]
    part = "%s" %x[-4:] #store current participant number
    for sessions in glob.glob("%s/*" %x): #Store list of sessions for current participant
        #sessions = glob.glob("%s/*" %x)[0]
        for files in glob.glob("%s/*" %sessions):
            #files = glob.glob("%s/*" %sessions)[0]
            current_session = files[68:-30]
            file = open(files, 'r')
            
            emotion = int(float(file.readline())) #emotions are encoded as a float, readline as float, then convert to integer.
            
            sourcefile_emotion = glob.glob("D:\\Training\\Source_Code\\Python\\MachineLearning\\Project\\cohn-kanade-images\\%s\\%s\\*" %(part, current_session))[-1] #get path for last image in sequence, which contains the emotion
            sourcefile_neutral = glob.glob("D:\\Training\\Source_Code\\Python\\MachineLearning\\Project\\cohn-kanade-images\\%s\\%s\\*" %(part, current_session))[0] #do same for neutral image
            
            dest_neut = "D:\\Training\\Source_Code\\Python\\MachineLearning\\Project\\Emotion_Classified\\neutral\\%s" %sourcefile_neutral[83:] #Generate path to put neutral image
            dest_emot = "D:\\Training\\Source_Code\\Python\\MachineLearning\\Project\\Emotion_Classified\\%s\\%s" %(emotions[emotion], sourcefile_emotion[83:]) #Do same for emotion containing image
            
            copyfile(sourcefile_neutral, dest_neut) #Copy file
            copyfile(sourcefile_emotion, dest_emot) #Copy file


# Format emotion images
path = 'D:\\Training\\Software\\Shared_Libraries\\openCV\\'
faceDet = cv2.CascadeClassifier(path+"haarcascade_frontalface_default.xml")
faceDet2 = cv2.CascadeClassifier(path+"haarcascade_frontalface_alt2.xml")
faceDet3 = cv2.CascadeClassifier(path+"haarcascade_frontalface_alt.xml")
faceDet4 = cv2.CascadeClassifier(path+"haarcascade_frontalface_alt_tree.xml")

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Define emotions

def detect_faces(emotion):
    print "Checking " + emotion
    files = glob.glob("D:\\Training\\Source_Code\\Python\\MachineLearning\\Project\\Emotion_Classified\\%s\\*" %emotion) #Get list of all images with emotion

    filenumber = 0
    for f in files:
        frame = cv2.imread(f) #Open image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convert image to grayscale
        
        #Detect face using 4 different classifiers
        face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face2 = faceDet2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face3 = faceDet3.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face4 = faceDet4.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)

        #Go over detected faces, stop at first detected face, return empty if no face.
        if len(face) == 1:
            facefeatures = face
        elif len(face2) == 1:
            facefeatures == face2
        elif len(face3) == 1:
            facefeatures = face3
        elif len(face4) == 1:
            facefeatures = face4
        else:
            facefeatures = ""
        
        #Cut and save face
        for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing face
            #print "face found in file: %s" %f
            gray = gray[y:y+h, x:x+w] #Cut the frame to size
            
            try:
                out = cv2.resize(gray, (350, 350)) #Resize face so all images have same size
                cv2.imwrite("D:\\Training\\Source_Code\\Python\\MachineLearning\\Project\\Emotion_Classified_Transformed\\%s\\%s.jpg" %(emotion, filenumber), out) #Write image
            except:
                print "Error with " + emotion
                pass #If error, pass file
        filenumber += 1 #Increment image number

for emotion in emotions: 
    detect_faces(emotion) #Call functiona