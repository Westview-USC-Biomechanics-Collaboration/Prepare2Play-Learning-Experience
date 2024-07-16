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


@app.route('/graph')
def graph():
    # Load your CSV file
    df = pd.read_csv('data/bjs_lr_DE_for01_Raw_Data - bjs_lr_DE_for01_Raw_Data.csv')

    # Select subset of data for plotting
    timex = df.iloc[18:12000, 0].astype(float).tolist()
    forcex = df.iloc[18:12000, 1].astype(float).tolist()
    forcey = df.iloc[18:12000, 2].astype(float).tolist()

    # Find the start of the force increase
    start_index = find_force_increase_start(forcex)

    # Subset the data starting from the force increase
    timex_subset = timex[start_index:]
    forcex_subset = forcex[start_index:]
    forcey_subset = forcey[start_index:]

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


if __name__ == '__main__':
    app.run(debug=True)
