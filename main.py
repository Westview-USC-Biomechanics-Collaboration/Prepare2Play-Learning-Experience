from flask import Flask, render_template
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/graph')
def graph():
    # Load your CSV file
    df = pd.read_excel('data\Trimmed of bjs_lr_DE_for01_Raw_Data (2).xlsx', skiprows=19, usecols=[0, 1, 2, 3, 10, 11, 12],
                       names=["time (s)", "Fx1", "Fy1", "Fz1", "Fx2", "Fy2", "Fz2"], header=0,
                       dtype={'time (s)': float, 'Fx1': float, 'Fy1': float, 'Fz1': float, 'Fx2': float, 'Fy2': float, 'Fz2': float})

    # Data points for plotting
    timex_subset = df.iloc[:, 0].astype(float).tolist()
    forcex_subset = savgol_filter(df.iloc[:, 1].astype(float).tolist(), 51, 3).tolist()  
    forcey_subset = savgol_filter(df.iloc[:, 2].astype(float).tolist(), 51, 3).tolist()
    forcez_subset = savgol_filter(df.iloc[:, 3].astype(float).tolist(), 51, 3).tolist()
    forcex2_subset = savgol_filter(df.iloc[:, 4].astype(float).tolist(), 51, 3).tolist()
    forcey2_subset = savgol_filter(df.iloc[:, 5].astype(float).tolist(), 51, 3).tolist()
    forcez2_subset = savgol_filter(df.iloc[:, 6].astype(float).tolist(), 51, 3).tolist()
    
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
