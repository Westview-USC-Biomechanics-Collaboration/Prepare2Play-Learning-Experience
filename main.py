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
