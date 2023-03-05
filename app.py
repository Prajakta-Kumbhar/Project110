# import the opencv library
import cv2
  
# define a video capture object
vid = cv2.VideoCapture(0)

import numpy as np
import tensorflow as tf
model = tf.keras.models.load_model("keras_model.h5")
  
while(True):
      
    # Capture the video frame by frame
    ret, frame = vid.read()
    image= cv2.resize(frame,(224,224))
    testimg= np.array(image,dtype=np.float32)
    testimg=np.expand_dims(testimg,axis=0)
    normalisedimg= testimg/255.0
    prediction = model.predict(normalisedimg)
    print(prediction)
  
    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # Quit window with spacebar
    key = cv2.waitKey(1)
    
    if key == 32:
        break
  
# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()