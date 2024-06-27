import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

# Initialize the Dash app
app = Dash(__name__)

# Specify the maximum number of rows to read
max_rows_to_read = 10  # Example: Limit to 1000 rows

# Load the data from the CSV file, starting from row 20, and assign column names
# Only select and name the columns you need
# Adjust 'usecols' and 'names' according to your CSV structure
df = pd.read_csv('data.csv', skiprows=19, usecols=[0, 1, 2, 3], names=["time (s)", "Fx", "Fy", "Fz"], header=0, nrows=max_rows_to_read,
                 dtype={'time (s)': float, 'Fx': float, 'Fy': float, 'Fz': float})

# Print first few rows of the DataFrame to inspect
print(df.head())

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
        range_x=[0, 0.5],  # Example: Limit x-axis range from 0 to 10
        range_y=[-1.5, 1.5],  # Example: Limit y-axis range from -5 to 5
        # mode='lines+markers'  # Change the mode to line+markers
    )

    frames = [dict(data=[dict(x=df["time (s)"][:k + 1], y=df[selected_force][:k + 1])]) for k in range(len(df))]
    figure.update(frames=frames, overwrite=True)


    return figure

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
