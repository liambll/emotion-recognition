# Author: LB
# Main program for live demo
import numpy as np
import cv2
import time
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout
import inception_v3 as inception

def load_model():
    N_CLASSES = 3
    # Start with an Inception V3 model, not including the final softmax layer.
    base_model = inception.InceptionV3(weights='imagenet')
    print 'Loaded Inception model'

    # Add on new fully connected layers for the output classes.
    x = Dense(32, activation='relu')(base_model.get_layer('flatten').output)
    x = Dropout(0.5)(x)
    predictions = Dense(N_CLASSES, activation='softmax', name='predictions')(x)
    model = Model(input=base_model.input, output=predictions)
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.load_weights('inception_happy_neutral_surprise.h5')
    return model

def show_webcam(mirror=False,camSource=0):
    IMSIZE = (299, 299)
    model = load_model()
    
    #Set up Camera
    font = cv2.FONT_HERSHEY_SIMPLEX
    cam = cv2.VideoCapture(camSource)
    
    #emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Emotion list
    #emotions = ["neutral", "anger", "disgust", "happy", "surprise"]
    #emotions = ["anger", "disgust", "fear", "happy", "sad", "surprise","neutral"]
    emotions = ["happy", "neutral", "surprise"]
    
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
            if mirror:
                img = cv2.flip(img, 1)
        
            # Detect Face
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            clahe_image = clahe.apply(gray)
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
                out = clahe_image[y:y+h, x:x+w] #Cut the frame to size
                cv2.imwrite('temp.jpg',out)
                
                img2 = image.load_img('temp.jpg', target_size=IMSIZE)
                x = image.img_to_array(img2)
                x = np.expand_dims(x, axis=0)
                x = inception.preprocess_input(x)
                pred = np.argmax(model.predict(x))
                #print pred
                preds[pred] = preds[pred] + 1
            
        #output predicted emotion on camera
        if np.max(preds) > 0:
            expression = emotions[np.argmax(preds)]
        else:
            expression = "No face Found"
        
        cv2.putText(img,expression,(10,40), font, 1,(255,255,255),2)
        cv2.imshow('my webcam', img)
        #time.sleep(0.15)
   
        if cv2.waitKey(1) == 27: # esc to quit
            break  
            
    cv2.destroyAllWindows()
    cam.release()
    
def main():
    show_webcam(mirror=False,camSource=0) #Front camera
    
if __name__ == '__main__':
    main()
    
