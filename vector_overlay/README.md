# Vector overlay & stick figure

This is the vector overlay folder, you can use scripts inside for visual post-processing

## Skeleton overlay/stick figure
- run `Skeletonoverlay.py`
- change the path to your video in the `video_path` variable

## Vector Overlay (quick start)
- run `vector-overlay-skeleton.py`
- scroll down to the last few lines
- uncomment the code for the top view or side view
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
