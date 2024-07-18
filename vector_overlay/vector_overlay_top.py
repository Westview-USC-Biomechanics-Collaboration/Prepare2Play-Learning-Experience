

# Paths
topview = "C:\\Users\\16199\\Documents\\GitHub\\Prepare2Play-Learning-Experience-3\\data\\gis_lr_CC_top_vid03.mp4"
forcedata_path = "../data/gis_lr_CC_for03_Raw_Data.xlsx"
output_name = "C:\\Users\\16199\\Documents\\GitHub\\Prepare2Play-Learning-Experience-3\\outputs\\" + topview[-23:-4] + "_vector_overlay.mp4"

def VectorOverlay(videopath, forcedata_path, filename):
    # import
    import cv2
    import pandas as pd
    from vector_overlay import contact_point
    # Load force data
    forcedata = pd.read_excel(forcedata_path, skiprows=1)

    # Open video
    cap = cv2.VideoCapture(videopath)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate skip rows based on data and frame count
    skiprows = int(forcedata.shape[0] / frame_count)

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    count_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't find frame")
            break

        row_to_use = 18 + int(count_frame * skiprows)
        force_row = forcedata.iloc[row_to_use]

        # Function to draw arrow and annotate contact and endpoint
        def drawArrow(contact_point, end_point, frame, Fx, Fy, contactpoint2, endpoint2):
            # Draw circles at the contact points
            cv2.circle(frame, center=contact_point, radius=7, color=(255, 0, 0), thickness=-1)
            cv2.circle(frame, center=contactpoint2, radius=7, color=(255, 0, 0), thickness=-1)
            # cv2.circle(frame, center=[457, 643], radius=5, color=(255, 0, 255))
            # Draw arrowed lines from contact points to end points
            cv2.arrowedLine(frame, contact_point, end_point, (0, 0, 255), thickness=2)
            cv2.arrowedLine(frame, contactpoint2, endpoint2, (0, 255, 0), thickness=2)
            # Annotate contact points and end points on the frame
            # cv2.putText(frame, f"Contact Point: {contact_point}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # cv2.putText(frame, f"Endpoint: {end_point}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # cv2.putText(frame, f"Fx: {Fx}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # cv2.putText(frame, f"Fy: {Fy}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # cv2.putText(frame, f"angle_to_use: {angle_to_use_1}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # cv2.putText(frame, f"vector1_angle: {vector1_angle}", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # cv2.putText(frame, f"a1_coords: {a1_coords}", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # cv2.putText(frame, f"b1_coords: {b1_coords}", (10, 310), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # cv2.putText(frame, f"row: {force_row[5], force_row[6]}", (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return frame

        # Find contact point and endpoint
        contactpoint, endpoint, Fx1, Fy1, angle_to_use_1, vector1_angle, a1_coords, b1_coords, contactpoint2, endpoint2 = contact_point.find_contact_top(locationin=[[249, 626], [977, 614]], forcedata=force_row)

        def flip(lst, frame_height):
            if len(lst) > 1:
                lst[1] = frame_height - lst[1]
                return lst
            else:
                raise ValueError("List does not have enough elements to flip the second one.")

        # Draw annotations on the frame
        annotated_frame = drawArrow(contactpoint, endpoint, frame, Fx1, Fy1, contactpoint2, endpoint2)

        # Display the annotated frame
        cv2.imshow('Annotated Frame', annotated_frame)

        # Write to output video
        out.write(annotated_frame)

        # Check for user input to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        count_frame += 1

    # Release video capture and writer objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Call the function
# VectorOverlay(topview, forcedata_path, output_name)
