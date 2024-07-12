from flask import Flask, render_template
import pandas as pd
import numpy as np

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/graph')
def graph():
    # Load your CSV file
    df = pd.read_csv('data/bjs_lr_DE_for01_Raw_Data - bjs_lr_DE_for01_Raw_Data.csv')

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
                           
if __name__ == '__main__':
    app.run(debug=True)
