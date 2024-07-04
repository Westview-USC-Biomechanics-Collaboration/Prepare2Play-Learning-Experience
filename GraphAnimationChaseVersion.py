import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
TotalFrames = 60
# Initialize the Dash app
app = Dash(__name__)

# Specify the maximum number of rows to read
max_rows_to_read = 1000  # Example: Limit to 100 rows for simplicity

# Load the data from the CSV file, starting from row 20, and assign column names
df = pd.read_csv('data/data.csv', skiprows=19, usecols=[0, 1, 2, 3], names=["time (s)", "Fx", "Fy", "Fz"], header=0, nrows=max_rows_to_read,
                 dtype={'time (s)': float, 'Fx': float, 'Fy': float, 'Fz': float})

# Define the app layout
app.layout = html.Div([
    html.H4('Animated Force Components Over Time'),
    dcc.Graph(id="animation-graph"),
    html.P("Select a force component:"),
    # I can add script here
    dcc.RadioItems(
        id='force-component-selection',
        options=[
            {'label': 'Fx', 'value': 'Fx'},
            {'label': 'Fy', 'value': 'Fy'},
            {'label': 'Fz', 'value': 'Fz'}
        ],
        value='Fx',
    ),
    dcc.Interval(  # <--- Difference 1
        id='interval-component',
        interval=16,  # Interval in milliseconds <--- Difference 2
        n_intervals=0,
        disabled= True

    ),
])

# Define the callback to update the graph
@app.callback(
    Output("animation-graph", "figure"),
    [Input("force-component-selection", "value"),
     Input("interval-component", "n_intervals")]  # <--- Difference 3
)
def update_graph(selected_force, n_intervals):  # <--- Difference 3
    frames = [
        go.Frame(
            data = [
                go.Scatter(
                    x=df["time (s)"][:(k + 1) * (int(len(df["time (s)"]) / TotalFrames))],  # <--- Difference 4
                    y=df[selected_force][:(k + 1) * (int(len(df[selected_force]) / TotalFrames))],  # <--- Difference 4
                    mode='lines',
                    fillcolor='rgba(0, 100, 80, 0.2)'
                )
            ],
            name=str(k)
        ) for k in range(TotalFrames)  # <--- Difference 5
    ]

    figure = go.Figure(
        data=[
            go.Scatter(
                x=df["time (s)"],
                y=df[selected_force],
                mode='lines',
                fill='tozeroy',
                fillcolor='rgba(0, 100, 80, 0.2)'
            )
        ],
        layout=go.Layout(
            title=f'{selected_force} vs Time',
            xaxis=dict(title='Time (s)', range=[df["time (s)"].min(), df["time (s)"].max()]),
            yaxis=dict(title='Force (N)', range=[df[selected_force].min(), df[selected_force].max()]),
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 100, "redraw": False},  # <--- Difference 6
                                        "fromcurrent": True, "transition": {"duration": 0}}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                          "mode": "immediate",
                                          "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                'direction': 'left',
                'pad': {'r': 10, 't': 87},
                'showactive': False,
                'type': 'buttons',
                'x': 0.1,
                'xanchor': 'right',
                'y': 0,
                'yanchor': 'top'
            }],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 20},
                    "prefix": "Time:",
                    "visible": True,
                    "xanchor": "right"
                },
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [
                            [str(k)],
                            {"frame": {"duration": 0, "redraw": False},
                             "mode": "immediate",
                             "transition": {"duration": 0}}
                        ],
                        "label": f"{df['time (s)'][k * int(len(df) / TotalFrames)]:.3f}",  # <--- Difference 7
                        "method": "animate"
                    } for k in range(TotalFrames)  # <--- Difference 5
                ]
            }]
        ),
        frames=frames
    )

    return figure

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
