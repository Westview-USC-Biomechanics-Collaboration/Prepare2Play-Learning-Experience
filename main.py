<<<<<<< HEAD
from flask import Flask, render_template
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

app = Flask(__name__)
=======
from flask import Flask, send_from_directory, render_template, request, redirect, url_for, jsonify
import pandas as pd
import numpy as np
import os
# from syncing.sync import VideoSync

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'upload'

# Options for dropdown menu
data = {
    'Basketball': ['Jumpshot', 'Free Throw'],
    'Soccer': ['Corner Kick', 'Goal Kick']
}

@app.route('/about')
def about():
    return render_template('about.html')

>>>>>>> main

@app.route('/')
def index():
    return render_template('index.html')

<<<<<<< HEAD
=======

@app.route('/dropdown', methods=['POST', 'GET'])
def dropdown():
    sport = list(data.keys())
    selected_sport = None
    movements = []
    # If we user selected from the dropdown it'll update our page1w
    if request.method == 'POST':
        # Recieves data from the form created in MovementDropDown.html
        selected_sport = request.form.get('sport')
        # Opens the data dictionary for all options under the sport they picked
        movements = data.get(selected_sport, [])
        # Use lines below once we know where to take the user after they select a movement
        # if 'movements' in request.form:
        #     selected_movement = request.form.get('movements')
        # return redirect(url_for('another_route', sport=selected_sport, movement=selected_ movement))
        return render_template('MovementDropDown.html', sports=sport, movements=movements,
                               selected_sport=selected_sport)
    else:
        # Handle GET request (initial page load)
        return render_template('MovementDropDown.html', sports=sport, movements=movements,
                               selected_sport=selected_sport)
    
@app.route('/SportsTemplate', methods=['POST', 'GET'])
def sports_template():
    if request.method == 'POST':
        video_file = request.files['video_file']
        if video_file is not None:
            # Get the original filename
            filename = video_file.filename
            # Save the uploaded file to a temporary location
            video_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # NEED TO ADD POST PROCESSING STUFF HERE BEFORE SENDING THE FILE BACK TO SPORTS TEMPLATE
            # Render the SportsTemplate.html template again with the uploaded video
            return render_template('SportsTemplate.html', video_file=filename, uploaded=True, video_url=url_for('uploaded_file', filename=filename))
        else:
            return redirect(url_for('sports_template'))
    else: 
        return render_template('SportsTemplate.html', uploaded=False)

# Sends the file that is uploaded to the SportsTemplate 
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

>>>>>>> main
@app.route('/graph')
def graph():
    # Load your CSV file
    df = pd.read_excel("C:\\Users\\16199\Desktop\\vector_overlay\Chase\\Trimmed of bcp_lr_CC_for02_Raw_Data.xlsx", skiprows=19, usecols=[0, 1, 2, 3, 10, 11, 12],
                       names=["time (s)", "Fx1", "Fy1", "Fz1", "Fx2", "Fy2", "Fz2"], header=0,
                       dtype={'time (s)': float, 'Fx1': float, 'Fy1': float, 'Fz1': float, 'Fx2': float, 'Fy2': float, 'Fz2': float})


    timeMotionStarts = 3.6 #edit to where motion starts
    forceatStart = 0 #this value will need to be different for each force direction
    # Data points for plotting
    rows = df.shape[0]

    time = []
    fx1 = []
    fy1 = []
    fz1 = []
    fx2 = []
    fy2 = []
    fz2 = []

    BW = max(df.iloc[0][3], df.iloc[0][6])
    for i in range(rows):
        time.append(df.iloc[i][0]-df.iloc[0][0])
        fx1.append(df.iloc[i][1]/BW)
        fy1.append(df.iloc[i][2] / BW)
        fz1.append(df.iloc[i][3] / BW)
        fx2.append(df.iloc[i][4] / BW)
        fy2.append(df.iloc[i][5] / BW)
        fz2.append(df.iloc[i][6] / BW)

    timex_subset = list(time)
    forcex_subset = fx1
    forcey_subset = fy1
    forcez_subset = fz1
    forcex2_subset = fx2
    forcey2_subset = fy2
    forcez2_subset = fz2
    
    integral_forcex = np.trapz(forcex_subset, timex_subset) - timeMotionStarts * forceatStart
    integral_forcey = np.trapz(forcey_subset, timex_subset) - timeMotionStarts * forceatStart
    integral_forcez = np.trapz(forcez_subset, timex_subset) - timeMotionStarts * forceatStart
    integral_forcex2 = np.trapz(forcex2_subset, timex_subset) - timeMotionStarts * forceatStart
    integral_forcey2 = np.trapz(forcey2_subset, timex_subset) - timeMotionStarts * forceatStart
    integral_forcez2 = np.trapz(forcez2_subset, timex_subset) - timeMotionStarts * forceatStart

    formatted_integral_forcex = f"{integral_forcex:.2f}"
    formatted_integral_forcey = f"{integral_forcey:.2f}"
    formatted_integral_forcez = f"{integral_forcez:.2f}"
    formatted_integral_forcex2 = f"{integral_forcex2:.2f}"
    formatted_integral_forcey2 = f"{integral_forcey2:.2f}"
    formatted_integral_forcez2 = f"{integral_forcez2:.2f}"

    return render_template('graph.html',
                           num_row = rows,
                           forcex=forcex_subset, 
                           timex=timex_subset,
                           forcey=forcey_subset,
                           forcez=forcez_subset,
                           integral_forcex=formatted_integral_forcex,
                           integral_forcey=formatted_integral_forcey,
                           integral_forcez=formatted_integral_forcez,
                           additional_forcex=forcex2_subset,
                           additional_forcey=forcey2_subset,
                           additional_forcez=forcez2_subset,
                           integral_forcex2=formatted_integral_forcex2,
                           integral_forcey2=formatted_integral_forcey2,
                           integral_forcez2=formatted_integral_forcez2)

if __name__ == '__main__':
    app.run(debug=True)
