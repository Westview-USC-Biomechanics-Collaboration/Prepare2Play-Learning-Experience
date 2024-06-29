from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/graph')
def graph():
    # Load your CSV file
    df = pd.read_csv('data/SM_SbS_02_Raw_Data - SM_SoftballSwing_Trial2_Raw_Data.csv')

    # Select subset of data for plotting
    timex_subset = df.iloc[18:30, 0].tolist() # This data is limited to 11 data points for the Force and Time
    forcex_subset = df.iloc[18:30, 1].tolist()

    timey_subset = df.iloc[18:30, 0].tolist() # This data is limited to 11 data points for the Force and Time
    forcey_subset = df.iloc[18:30, 2].tolist()

    return render_template('graph.html', 
                           forcex=forcex_subset, 
                           timey=timey_subset,
                           forcey=forcey_subset, 
                           timex=timex_subset)
if __name__ == '__main__':
    app.run(debug=True)