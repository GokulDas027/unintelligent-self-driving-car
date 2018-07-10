# unintelligent-self-driving-car
It detects road lanes and keep to the road. A self rolling system that cares only about road lanes.Hobby Project

caution: This may be totally stupid in times :D

It just GrabScreen, find lanes and tries to navigate through the center of the lanes. 
The directkeys.py makes key entries based on the, screen grabbed by grabscreen.py processed by screen_process.py .

To find lanes : Original image -> Grayscaled image -> Canny Edge Detection -> Gaussian Blur -> Region of Interest Selection -> Hough Line Detection -> Draw detected lanes on Original image.
Working : if: lane not found in any one direction turn to that direction else: go forward (sorry he didn't learned to brake).

Dependencies : Numpy, OpenCV, Times, Ctypes, pyWin32

Just as a hobby project,could be optimised for better work but you can't invent bulb by modifying candle. 
Thanks udacity-self-driving-car free preview ;) and Sendtex. Obiviously Stack overflow and Google too. 
This is important because this is the first self project done by me ;) Need to learn more, code more. 
But i am excited as of now, someday to look back.
