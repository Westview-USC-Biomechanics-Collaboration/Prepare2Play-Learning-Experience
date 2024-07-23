## RULES:
- Must make changes on a separate branch and open a pull request before merging to main
- All pull requests require 1 approval from a designated "code owner" (TBD) before it can be merged 
- Delete your branches that are no longer in use
- Please make your commit messages and pull request descriptions detailed and descriptive of what changes you made

## Installation:
- Download Python 3.11.X from **https://www.python.org/downloads/**
- Download git from **https://git-scm.com/download/**
- Run the program and select "create .venv file" and select the Python version on your machine
- In terminal run `pip install -r requirements.txt` (`pip install -r requirements.txt` also works)

## Run the website
- Run `python3 main.py` to run the website 
- Access the program at **https://localhost:5000** or **http://127.0.0.1:5000** (Depending on your machine) to test local changes

##  Process your video with a vector overlay
- Make sure your video and force file are trimmed to exactly when the ball hits the forceplate
- Place your trimmed video (`.mp4` or `.mov`) and your trimmed force data (`.csv`) in the `data` folder.
- In `vector_overlay/vector-overlay-skeleton.py` update the `side_view`, `forcedata_path`, and `top_view` variables with the appropriate paths `data/[FILENAME]`
- At the bottom of the file, there are two comments, sideview, and topview (comments are marked with a # at the beginning of them). Depending on what code you want to run, uncomment those lines.
  - EX: If you want the sideview, leave the code under `#sideview` uncommented, and the code under `#topview` should be commented so it doesn't run.
- Now, in your terminal, make sure you are in the project, and run `python3 vector_overlay/vector-overlay-skeleton.py`
- NOTE: If your video was made before the tape was place, you will have to manually select the 8 corners for the side view in this order `[INSERT CORNER ORDER HERE]`
- Navigate to `outputs/` and find the file with the appropriate naming convention for your movement
- If you find any issues, please contact the software team! We are happy to help

## This is the vector overlay branch

## Contributors:
- Chase Chen
- Ayaan Irshad
- Deren Erdem

