import threading
import cv2
import time
from Util.ballDropDetect import ballDropDetect

capture = cv2.VideoCapture('/home/chaser/Downloads/tss_rl_JG_vid02.mov')

# Correct way to pass capture as a single argument in a tuple
ballDetectThread = threading.Thread(target=ballDropDetect, args=(capture,))  
ballDetectThread.start()

for i in range(10):
    print(i + 1)
    time.sleep(1)

ballDetectThread.join()  # Wait for the thread to finish
