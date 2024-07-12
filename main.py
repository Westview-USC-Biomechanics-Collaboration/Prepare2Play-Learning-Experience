from flask import Flask, render_template
import pandas as pd
import numpy as np

app = Flask(__name__)

def find_force_increase_start(force, threshold=4.5):
    for i, f in enumerate(force):
        if f > threshold:
            return i
    return 0

@app.route('/')
def index():
    return render_template('index.html')

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
