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

#HoughLine Function
def hough_lines(img, rho=1, theta=np.pi/180, threshold=180, min_line_len=20, max_line_gap=15):
    #`img` should be the output of a Canny transform.        
    #Returns an image with hough lines drawn.
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

#Functions to draw average & extrapolate lines
def compute_line(m, b, x1, x2):
    return int(x1), int(m*x1 + b), int(x2), int(m*x2 + b)

def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    if lines is None or len(lines) == 0 :
        return
    left_m_sum = 0
    left_b_sum = 0
    num_of_left_lines = 0    
    right_m_sum = 0
    right_b_sum = 0
    num_of_right_lines = 0    
    min_slope = 0.4
    max_slope = 2
    
    for line in lines:
        for x1,y1,x2,y2 in line:           
            #lines perpenticular to screen is avoided
            if x1 == x2:
                continue 
                
            m = ((y2-y1)/(x2-x1)) #eqn to find slope of line
            b = y1 - m*x1
            
            #filtering to remove noises
            if abs(m) < min_slope or abs(m) > max_slope:
                continue
                
            if m > 0:
                left_m_sum += m
                left_b_sum += b
                num_of_left_lines += 1
            else:
                right_m_sum += m
                right_b_sum += b
                num_of_right_lines +=1
                
    avg_lines = []
    width = img.shape[1]
    
    if num_of_left_lines > 0:
        left_m = left_m_sum / float(num_of_left_lines)
        left_b = left_b_sum / float(num_of_left_lines)
        
        left_line = [compute_line(left_m, left_b, width, width*0.52)]
        avg_lines.append(left_line)
        
    if num_of_right_lines > 0:
        right_m = right_m_sum / float(num_of_right_lines)
        right_b = right_b_sum / float(num_of_right_lines)
        
        right_line = [compute_line(right_m, right_b, -width*.55, width*0.48)]
        avg_lines.append(right_line)
               
    for averaged_line in avg_lines:
        for x1,y1,x2,y2 in averaged_line:
            cv2.line(img, (x1, y1), (x2, y2), color, 15)         
                
    print(avg_lines) #testing purpose

#Transparent drawing on image
def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    #initial_img * α + img * β + γ
    return cv2.addWeighted(initial_img, α, img, β, γ)

# Basic Processing of Image
def process_img(original_image):
    #Grayscaling image
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    #CannyEdge Detection
    processed_img =  cv2.Canny(processed_img, threshold1 = 75, threshold2=255)
    #Gaussian Blur
    processed_img = cv2.GaussianBlur(processed_img, (5, 5), 0)
    #Region of Interest Selection
    #vertices for selection of front view roi
    #vertices = np.array([[(0,600),(0,420),(310,300), (490, 300), (800,420),(800,600),(440,400),(360,400),(0,600)]], dtype=np.int32)
    #vertices for selection of long back cam view roi
    vertices = np.array([[(50,600),(310,260),(480,260), (720, 600), (590,600),(440,330),(355,330),(200,600)]], dtype=np.int32)
    processed_img = roi(processed_img, [vertices])
    #Hough : line detection, draw_lines() is called from inside hough_lines()
    processed_img = hough_lines(processed_img)
    #Draw lanes on original image
    processed_img = weighted_img(processed_img,original_image)

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




