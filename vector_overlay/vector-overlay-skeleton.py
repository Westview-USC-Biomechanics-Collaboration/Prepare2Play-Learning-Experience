#Import libraries
import cv2
import pandas as pd
# Where we import our python script
import contact_point


# initialization
topview = "C:\\Users\\16199\Documents\GitHub\Prepare2Play-Learning-Experience-3\data\gis_lr_CC_top_vid02.mp4"
sideview = "data/gis_lr_CC_vid02.mp4"
forcedata_path ="C:\\Users\\16199\Documents\GitHub\Prepare2Play-Learning-Experience-3\data\\gis_lr_CC_for02_Raw_Data.xlsx"
output_name = "C:\\Users\\16199\Documents\GitHub\Prepare2Play-Learning-Experience-3\outputs\\" + topview[-23:-4] + "_vector_overlay.mp4"
print(f"outputname: {output_name}")
# function

def VectorOverlay(videopath, forcedata_path, filename):
    # Process data
    forcedata = pd.read_excel(forcedata_path, skiprows=1)
    print(f"This is forcedata: {forcedata.head}")
    print("*******************************************")
    total_rows_columns = forcedata.shape
    print(f"This is totalrows: {total_rows_columns[0]}")
    # Testing code below
    input = [[457, 643], [978, 648]]
    example_input = [[int(x * 1) for x in sublist] for sublist in input]
    print(f"This is example_input: {example_input}")
    example_forcedata = forcedata.iloc[18]


    # Process video
    cap = cv2.VideoCapture(videopath)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    #skip rows:
    skiprows = int(total_rows_columns[0]/frame_count)
    print(f"Skipsrows = {skiprows}")

    # Define the codec and create VideoWriter object to save the annotated video
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                          (frame_width, frame_height))

    count_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't find frame")
            break

        row_to_use = 18 + int(count_frame * skiprows)
        force_row = forcedata.iloc[row_to_use]
        def drawArrow(contact_point, end_point, frame):
            # Resize the frame to a smaller size
            # frame_resized = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            cv2.circle(frame, center=[457, 643], radius= 5 , color = (255,0,0))
            # Draw a circle at the contact point
            cv2.circle(frame, center=contact_point, radius=7, color=(0, 0, 255),
                       thickness=-1)  # thickness=-1 fills the circle

            # Draw an arrowed line from contact point to end point
            cv2.arrowedLine(frame, contact_point, end_point, (0, 255, 0), thickness=2)

            return frame

        # Where our code start
        contactpoint, endpoint = contact_point.find_contact_top(locationin=example_input, forcedata=force_row)
        print(f"This is contactpoint = {contactpoint}")
        print(f"This is endpoint = {endpoint}")
        annotated_frame = drawArrow(contactpoint,endpoint,frame)


        cv2.imshow('Annotated Frame', annotated_frame)

        width, height = 800, 600  # Desired window size
        cv2.resizeWindow('Annotated Frame', width, height)

        out.write(annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        count_frame += 1

    # Release video capture and writer objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()




# call funciton
VectorOverlay(topview, forcedata_path, output_name)

