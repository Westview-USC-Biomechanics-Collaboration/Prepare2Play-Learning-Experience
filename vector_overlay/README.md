# Vector overlay & stick figure

This is the vector overlay folder, you can use scripts inside for visual post-processing
Please use `Skeletonoverlay.py` for stick figure (This program doesn't work well in top view)
Please use `vector-overlay-skeleton.py` for vector overlay(This is the program that gets the force on force plate and visualize it)

# Seting paths
- if your os is Windows, use "\\"
- if your os is ios, use "/"
- if the relative path doesn't work, try the absolute path(e.x. "C:\\Users\\16199\\Documents\\GitHub\\Prepare2Play-Learning-Experience-3\\data\\ajp")

## Skeleton overlay/stick figure (quick start)
- change the path to your video in the `video_path` variable
- run `Skeletonoverlay.py`

## Vector Overlay (quick start)
- run `vector-overlay-skeleton.py`
- scroll down to the last few lines
- change the path to the file
- run the program

##  Process your video with a vector overlay (detailed)
- Make sure your video and force file are trimmed to exactly when the ball hits the forceplate
- Place your trimmed video (`.mp4` or `.mov`) and your trimmed force data (`.xlsx`or `.csv`) in the `data` folder.
- In `vector_overlay/vector-overlay-skeleton.py` update the `side_view`, `forcedata_path`, and `top_view` variables with the appropriate paths `data/[FILENAME]`
- At the bottom of the file, there are two comments, sideview, and topview (comments are marked with a # at the beginning of them). Depending on what code you want to run, uncomment those lines.
  - EX: If you want the sideview, leave the code under `#sideview` uncommented, and the code under `#topview` should be commented so it doesn't run.
- Now, in your terminal, make sure you are in the project, and run `python3 vector_overlay/vector-overlay-skeleton.py`
- NOTE: If your video was made before the tape was place, you will have to manually select the 8 corners for the side view in this order `[INSERT CORNER ORDER HERE]`
- Navigate to `outputs/` and find the file with the appropriate naming convention for your movement
- If you find any issues, please contact the software team! We are happy to help

# Contributors
1. Ayaan Irshad
2. Chase Chen
3. Aarav Yadav
4. Deren Erdem
