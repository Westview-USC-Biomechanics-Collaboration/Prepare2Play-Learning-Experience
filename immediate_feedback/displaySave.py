import cv2 as cv
from vector_overlay import vector_overlay_skeleton


# With a set of coordinates, display a frame with annotated arrows (long view), and save the frames

def render(frame, xf1, xf2, yf1, yf2, px1, px2, py1, py2):
    data = "/"  # Put some valid path to any data file ( we don't need it tho )
    v = vector_overlay_skeleton.VectorOverlay(data_path=data)
    v.drawArrows(frame, xf1, xf2, yf1, yf2, px1, px2, py1, py2)


cam = cv.VideoCapture(1)
# Camera code copied from https://github.com/Westview-USC-Biomechanics-Collaboration/Prepare2Play-Learning-Experience/blob/chase-immediate-feed-back/vector_overlay/camera.py
if not cam.isOpened():
    print("Error: Could not open camera.")
    exit()

# Loop to continuously capture frames
frameNum = 0
while True:
    frameNum+=1
    # Capture a frame
    ret, frame = cam.read()

    # Check if frame is captured successfully
    if not ret:
        print("Error: Could not read frame.")
        break

    # Display the frame
    render(frame, 0, 0, 0, 0, 0, 0, 0, 0)
    # cv.imwrite("saved_frames/frame_" + frameNum, frame) # Writes every frame to memory (a lot of files, be careful if you uncomment it)
    cv.imshow('Webcam', frame)

    # Break the loop on pressing 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cam.release()
cv.destroyAllWindows()
