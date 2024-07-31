from flask import Flask, render_template
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/anglegraph')
def graph():
    # Load your CSV file
    df = pd.read_csv('data/data.csv')

    # Select subset of data for plotting
    angle1_subset = df.iloc[18:11803, 0].astype(float).tolist()
    angle2_subset = df.iloc[18:11803, 1].astype(float).tolist()
    
    return render_template('anglegraph.html', 
                           angle1=angle1_subset, 
                           angle2=angle2_subset
                        )

if __name__ == '__main__':
    app.run(debug=True)
