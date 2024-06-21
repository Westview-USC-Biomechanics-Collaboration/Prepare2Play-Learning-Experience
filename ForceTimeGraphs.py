from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('LandingPage.html')

@app.route('/graph')
def graph():
    # Load your CSV file
    df = pd.read_csv('data.csv')

    # Select subset of data for plotting
    forcex_subset = df.iloc[18:30, 0].tolist() # This data is limited to 11 data points for the Force and Time
    timey_subset = df.iloc[18:30, 1].tolist()

    return render_template('graph.html', forcex=forcex_subset, timey=timey_subset)

if __name__ == '__main__':
    app.run(debug=True)
