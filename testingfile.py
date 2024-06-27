import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

# Initialize the Dash app
app = Dash(__name__)

# Specify the maximum number of rows to read
max_rows_to_read = 1000  # Example: Limit to 1000 rows

# Load the data from the CSV file, starting from row 20, and assign column names
df = pd.read_csv('data.csv', skiprows=19, names=["time (s)", "Fx", "Fy", "Fz"], header=0, nrows=max_rows_to_read)

# Define the app layout
app.layout = html.Div([
    html.H4('Animated Force Components Over Time'),
    dcc.Graph(id="animation-graph"),
    html.P("Select a force component:"),
    dcc.RadioItems(
        id='force-component-selection',
        options=[
            {'label': 'Fx', 'value': 'Fx'},
            {'label': 'Fy', 'value': 'Fy'},
            {'label': 'Fz', 'value': 'Fz'}
        ],
        value='Fx',
    ),
])

# Define the callback to update the graph
@app.callback(
    Output("animation-graph", "figure"),
    Input("force-component-selection", "value"))
def update_graph(selected_force):
    figure = px.scatter(
        df, x="time (s)", y=selected_force,
        animation_frame="time (s)", animation_group="time (s)",
        title=f'{selected_force} vs Time',
        labels={'time (s)': 'Time (s)', selected_force: 'Force (N)'},
        template='plotly_white',
        # mode='lines+markers'  # Change the mode to line+markers
    )
    return figure

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
