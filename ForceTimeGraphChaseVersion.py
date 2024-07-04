from flask import Flask, render_template
import pandas as pd
import plotly.express as px
import dash
from dash import html, dcc
from dash.dependencies import Input, Output

# Create Flask server
server = Flask(__name__)

# Create Dash app
app = dash.Dash(__name__, server=server, url_base_pathname='/TestGraph/')

# Specify column names
column_names = ["time", "forcex", "forcey", "forcez"]

# Read the CSV file and assign column names
# df = pd.read_csv('data.csv', names=column_names, header=None)

# Example data (replace with your actual data loading)
df = pd.read_csv('data/data.csv')
time_subset = df.iloc[18:18696, 0].tolist()
forcex_subset = df.iloc[19:18696, 1].tolist()
forcey_subset = df.iloc[19:18696, 2].tolist()
forcez_subset = df.iloc[19:18696, 3].tolist()


# Define Dash app layout
app.layout = html.Div([
    html.H4("force vs time"),
    html.P("Select data on y-axis:"),
    dcc.Dropdown(
        id='y-axis',
        options=[
            {'label': 'force x', 'value': 'force x'},
            {'label': 'force y', 'value': 'force y'},
            {'label': 'force z', 'value': 'force z'}
        ],
        value='force x'
    ),
    dcc.Graph(id="graph"),

# Back to Home button
    html.Br(),
    html.A('Back to Home', href='/'),
])


# Define the callback to update the graph
@app.callback(
    Output('graph', 'figure'),
    [Input('y-axis', 'value')]
)
def update_graph(selected_force):
    if selected_force == 'force x':
        y_data = forcex_subset
    elif selected_force == 'force y':
        y_data = forcey_subset
    elif selected_force == 'force z':
        y_data = forcez_subset

    figure = {
        'data': [{
            'x': time_subset,
            'y': y_data,
            'type': 'scatter',
            'mode': 'lines+markers',
            'fill': 'tozeroy',  # Fills area to the x-axis
            'fillcolor': 'rgba(0, 100, 80, 0.2)'  # Optional: Custom fill color with transparency
        }],
        'layout': {'title': f'{selected_force} vs time'}
    }

    return figure





# Define Flask route for the home page
@server.route('/')
def index():
    return render_template('index.html')


# Define Flask route for the Dash app
@server.route('/TestGraph/')
def render_dash():
    return app.index()


# Define another Flask route (if needed)
@server.route('/graph')
def graph():
    return render_template('graph.html',
                           forcex=forcex_subset,
                           forcey=forcey_subset,
                           forcez=forcez_subset,
                           time=time_subset)


# Run the Flask server
if __name__ == '__main__':
    server.run(debug=True)
