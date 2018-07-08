import numpy as np
from PIL import ImageGrab
import cv2
import time

# Selecting Region of interest
def roi(img, vertices):    
    #blank mask:
    mask = np.zeros_like(img)   
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, 255)
    #returning the image only where mask pixels are nonzero
    masked = cv2.bitwise_and(img, mask)
    return masked

# Basic Processing of Image
def process_img(original_image):
    #Grayscaling image
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    #CannyEdge Detection
    processed_img =  cv2.Canny(processed_img, threshold1 = 75, threshold2=255)
    #Selection
    imshape = original_image.shape

    #front view roi
    vertices = np.array([[(0,600),(0,420),(310,300), (490, 300), (800,420),(800,600),(440,400),(360,400),(0,600)]], dtype=np.int32)
    #long back cam view roi
    #vertices = np.array([[(50,600),(310,260),(480,260), (720, 600), (590,600),(440,330),(355,330),(200,600)]], dtype=np.int32)

    processed_img = roi(processed_img, [vertices])   
    return processed_img

# Reading Screen
def screen_read(): 
    last_time = time.time()
    while(True):
        # 800x600 windowed mode, at the top left position.
        # 40 px accounts for title bar. 
        printscreen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
        # to know the frames per second.
        print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()

        cv2.imshow('window',process_img(cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB)))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    
screen_read()




