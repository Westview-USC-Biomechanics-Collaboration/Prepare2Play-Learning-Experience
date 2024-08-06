# Vector overlay & stick figure

- This is the vector overlay folder, you can use scripts inside for visual post-processing
- Please use `stick_figure_COM.py` for stick figure (This program doesn't work well in top view)
- Please use `vector-overlay-skeleton.py` for vector overlay(This is the program that gets the force on force plate and visualize it)

# Seting paths
- if your os is Windows, use "\\"
- if your os is ios, use "/"
- if the relative path doesn't work, try the absolute path(e.x. "C:\\Users\\16199\\Documents\\GitHub\\Prepare2Play-Learning-Experience-3\\data\\ajp")

## Skeleton overlay/stick figure (quick start)
- change the path to your video in the `video_path` variable
- run `stick_figure_COM.py`

## Vector Overlay (quick start)
- run `vector-overlay-skeleton.py`
- scroll down to the last few lines
- change the path to the file
- run the program

##  Process your video with a vector overlay (detailed)
- Create a two new folders in the main directory called `outputs` and 'data'.
- Make sure your video and force file are trimmed to exactly when the ball hits the forceplate
- Place your trimmed videos (`.mp4') and your trimmed force data (`.xlsx`or `.csv`) in the `data/[NAME]` folder.
- NOTE: make sure after the initials on the video, you include the format, (i.e. `spk_lr_AI_long_vid02.mp4`, where AI are initials, and long is the format)
- In `vector_overlay/vector-overlay-skeleton.py`, update the path on line 424 with the name of your student (the same as the folder)
- Now, in your terminal, make sure you are in the project, and run `python3 vector_overlay/vector-overlay-skeleton.py`
- Select the order of the corners following this [diagram](https://github.com/Westview-USC-Biomechanics-Collaboration/Prepare2Play-Learning-Experience/blob/main/vector_overlay/vector-overlay-skeleton.py) (each orientation has the numbers on the corners in the order to select them)
- Navigate to `outputs/` and find the file with the appropriate naming convention for your movement. (*optional: set the path to your custom folder)
- If you find any issues, please contact the software team! We are happy to help

# Contributors
1. Ayaan Irshad
2. Chase Chen
3. Aarav Yadav
4. Deren Erdem
5. Jessie Bao
6. Ross Brown
7. Imri Fernandes
8. 8. James Guo
9. Rayyan Hussain
10. Nishk Shah
11. Breanna Thayillam
12. Christopher Yuan
