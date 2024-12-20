from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/graph')
def graph():
    # Load your CSV file
    #df = pd.read_csv('data/SM_SbS_02_Raw_Data - SM_SoftballSwing_Trial2_Raw_Data.csv')
    df = pd.read_csv('data\sbs_lr_SM_for02_Raw_Data_ - SM_SoftballSwing_Trial2_Raw_Data.csv')
    # Select subset of data for plotting
    # Row 18 - 29 and column 0 (the first one)
    forcex_subset = df.iloc[18:500, 0].tolist() # This data is limited to 11 data points for the Force and Time
    timey_subset = df.iloc[18:500, 1].tolist()

    return render_template('graph.html', forcex=forcex_subset, timey=timey_subset)

if __name__ == '__main__':
    app.run(debug=True)
