# Author: LB
# Main program for live demo
import numpy as np
import cv2
import time

def show_webcam(mirror=False,camSource=0):
    #Set up Camera
    font = cv2.FONT_HERSHEY_SIMPLEX
    cam = cv2.VideoCapture(camSource)
    
    #emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Emotion list
    #emotions = ["neutral", "anger", "disgust", "happy", "surprise"]
    #emotions = ["anger", "disgust", "fear", "happy", "sad", "surprise","neutral"]
    #emotions = ["disgust", "happy", "neutral", "surprise"]
    emotions = ["happy", "neutral", "surprise"]
    
    # Emotion Recognizer
    fishface = cv2.face.createFisherFaceRecognizer()
    #fishface = cv2.face.createEigenFaceRecognizer();
    #fishface = cv2.face.createLBPHFaceRecognizer();
    fishface.load('all_happy_neutral_surprise.xml')
    
    # Face Recognizer
    faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    #faceDet2 = cv2.CascadeClassifier(path+"haarcascade_frontalface_alt2.xml")
    #faceDet3 = cv2.CascadeClassifier(path+"haarcascade_frontalface_alt.xml")
    #faceDet4 = cv2.CascadeClassifier(path+"haarcascade_frontalface_alt_tree.xml")
    
    no_frames = 1
    while True:
        preds = [0] * len(emotions)
        #Capture image from camera
        for i in range(no_frames):
            ret_val, img = cam.read()
            
            # Detect Face
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            #clahe_image = clahe.apply(gray)
            clahe_image = gray
            face = faceDet.detectMultiScale(clahe_image, scaleFactor=1.1, minNeighbors=15, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)
            #face2 = faceDet2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
            #face3 = faceDet3.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
            #face4 = faceDet4.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
            if len(face) == 1:
                facefeatures = face
                #elif len(face2) == 1:
                #    facefeatures == face2
                #elif len(face3) == 1:
                #    facefeatures = face3
                #elif len(face4) == 1:
                #    facefeatures = face4
            else:
                facefeatures = ""        
        
            #If found face, predict corresponding emotion
            for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing face
                #print "face found in file: %s" %f
                gray = clahe_image[y:y+h, x:x+w] #Cut the frame to size
                out = cv2.resize(gray, (350, 350))
                #out = cv2.resize(gray, (350, 350))
                #predict emotion based on input img
                pred = fishface.predict(out)
                #print pred
                preds[pred] = preds[pred] + 1
            
        #output predicted emotion on camera
        if np.max(preds) > 0:
            expression = emotions[np.argmax(preds)]
        else:
            expression = "No face Found"
        
        if mirror:
            img_display = cv2.flip(img, 1)
        else:
            img_display = img
                
        cv2.putText(img_display,expression,(10,40), font, 1,(255,255,255),2)
        cv2.imshow('my webcam', img_display)
        #time.sleep(0.15)
   
        if cv2.waitKey(1) == 27: # esc to quit
            break  
            
    cv2.destroyAllWindows()
    cam.release()
    
def main():
    show_webcam(mirror=True,camSource=1) #Front camera
    
if __name__ == '__main__':
    main()
    
