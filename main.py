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
    df = pd.read_csv('data/Trimmed of bjs_lr_DE_for01_Raw_Data - bjs_lr_DE_for01_Raw_Data.csv')

    # Select subset of data for plotting
    timex_subset = df.iloc[18:11803, 0].astype(float).tolist()
    forcex_subset = df.iloc[18:11803, 1].astype(float).tolist()
    forcey_subset = df.iloc[18:11803, 2].astype(float).tolist()
    forcez_subset = df.iloc[18:11803, 3].astype(float).tolist() 
    forcex2_subset = df.iloc[18:11803, 10].astype(float).tolist()
    forcey2_subset = df.iloc[18:11803, 11].astype(float).tolist()
    forcez2_subset = df.iloc[18:11803, 12].astype(float).tolist() 

    integral_forcex = np.trapz(forcex_subset, timex_subset)
    integral_forcey = np.trapz(forcey_subset, timex_subset)
    integral_forcez = np.trapz(forcez_subset, timex_subset)
    integral_forcex2 = np.trapz(forcex2_subset, timex_subset)
    integral_forcey2 = np.trapz(forcey2_subset, timex_subset)
    integral_forcez2 = np.trapz(forcez2_subset, timex_subset)

    formatted_integral_forcex = f"{integral_forcex:.2f}"
    formatted_integral_forcey = f"{integral_forcey:.2f}"
    formatted_integral_forcez = f"{integral_forcez:.2f}"
    formatted_integral_forcex2 = f"{integral_forcex2:.2f}"
    formatted_integral_forcey2 = f"{integral_forcey2:.2f}"
    formatted_integral_forcez2 = f"{integral_forcez2:.2f}"

    
    return render_template('graph.html', 
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
