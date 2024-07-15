#Import libraries
import cv2
import pandas as pd

# initialization
topview = "data/gis_lr_CC_top_vid02.mp4"
sideview = "data/gis_lr_CC_vid02.mp4"
forcedata_path ="data/gis_lr_CC_for02_Raw_Data.xlsx"
output_name = topview[5:-4] + "_vector_overlay"
print(output_name)
# function

def VectorOverlay(videopath, forcedata_path, filename):
    forcedata = pd.read_excel(forcedata_path, skiprows=19)
    cap = cv2.VideoCapture(videopath)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object to save the annotated video
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                          (frame_width, frame_height))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't find frame")
            break








    # Release video capture and writer objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()




# call funciton
VectorOverlay(topview, forcedata_path, output_name)

