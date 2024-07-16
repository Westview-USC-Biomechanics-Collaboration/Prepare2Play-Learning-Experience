from flask import Flask, render_template, request, redirect, url_for, jsonify, escape
import pandas as pd
import numpy as np

# from syncing.sync import VideoSync

app = Flask(__name__)

# Options for dropdown menu
data = {
    'Basketball': ['Jumpshot', 'Free Throw'],
    'Soccer': ['Corner Kick', 'Goal Kick']
}

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/')
def index():
    return render_template('index.html')


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
    
@app.route('/SportsTemplate.html', methods=['POST', 'GET'])
def sports_template():
    return render_template('SportsTemplate.html')

@app.route('/graph')
def graph():
    # Load your CSV file
    df = pd.read_csv('data/SM_SbS_02_Raw_Data - SM_SoftballSwing_Trial2_Raw_Data.csv')

    # Select subset of data for plotting
    timex_subset = df.iloc[18:10000, 0].astype(float).tolist()
    forcex_subset = df.iloc[18:10000, 1].astype(float).tolist()
    forcey_subset = df.iloc[18:10000, 2].astype(float).tolist()
    integral_forcex = np.trapz(forcex_subset, timex_subset)
    integral_forcey = np.trapz(forcey_subset, timex_subset)
    formatted_integral_forcex = f"{integral_forcex:.2f}"
    formatted_integral_forcey = f"{integral_forcey:.2f}"

    # Render graph.html template and pass data to frontend
    return render_template('graph.html',
                           forcex=forcex_subset,
                           timex=timex_subset,
                           forcey=forcey_subset,
                           integral_forcex=formatted_integral_forcex,
                           integral_forcey=formatted_integral_forcey)


@app.route("/sync", methods=["GET", "POST"])
def syncVideo():
    videoPath = ""
    if request.method == "POST":
        # videoPath = "data/NS_SPU_01.mov"
        # csvPath = "data/NS_SPU_01_Raw_Data - NS_SurfPopUp_Trial1_Raw_Data.csv"
        #
        # sync = VideoSync(videoPath, csvPath)
        # videoPath = sync.syncSave()
        # print("Video Generated!")
        # videoPath = videoPath.lstrip("../static/")
        videoPath = "syncedVideo/sycnedVideo.mp4"
    return videoPath


if __name__ == '__main__':
    app.run(debug=True)
