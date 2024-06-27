import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objs as go

# Initialize the Dash app
app = Dash(__name__)

# Specify the maximum number of rows to read
max_rows_to_read = 1000  # Example: Limit to 1000 rows

# Load the data from the CSV file, starting from row 20, and assign column names
# Only select and name the columns you need
# Adjust 'usecols' and 'names' according to your CSV structure
df = pd.read_csv('data.csv', skiprows=19, usecols=[0, 1, 2, 3], names=["time (s)", "Fx", "Fy", "Fz"], header=0,
                 nrows=max_rows_to_read,
                 dtype={'time (s)': float, 'Fx': float, 'Fy': float, 'Fz': float})

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
    # Create initial figure layout
    figure = go.Figure()
    figure.update_layout(
        title=f'{selected_force} vs Time',
        xaxis_title='Time (s)',
        yaxis_title='Force (N)',
        template='plotly_white',
        showlegend=False,
    )

    # Add scatter trace for the initial data
    figure.add_trace(
        go.Scatter(
            x=df["time (s)"], y=df[selected_force],
            mode='lines+markers',  # Change mode to markers
            marker=dict(color='blue', size=10),
            name=selected_force
        )
    )

    # Update animation frames
    frames = []
    for i in range(len(df)):
        frame_data = go.Frame(
            data=[go.Scatter(
                x=df["time (s)"][:i + 1], y=df[selected_force][:i + 1],
                mode='lines+markers',
                marker=dict(color='blue', size=10),
                name=selected_force
            )],
            name=str(i)  # Frame name as string
        )
        frames.append(frame_data)

    # Add frames to the figure
    figure.frames = frames

    return figure


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
