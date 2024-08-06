import pandas as pd
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

TotalFrames = 100

# Initialize the Dash app
app = Dash(__name__)

# Load the data from the Excel file, starting from row 20, and assign column names
df = pd.read_excel('data/Trimmed of gis_lr_CC_for03_Raw_Data.xlsx', skiprows=19, usecols=[0, 1, 2, 3, 10, 11, 12],
                   names=["time (s)", "Fx1", "Fy1", "Fz1", "Fx2", "Fy2", "Fz2"], header=0,
                   dtype={'time (s)': float, 'Fx1': float, 'Fy1': float, 'Fz1': float, 'Fx2': float, 'Fy2': float, 'Fz2': float})

# Define the app layout
app.layout = html.Div([
    html.H4('Animated Force Components Over Time'),
    dcc.Loading(
        [dcc.Graph(id="animation-graph-plate1"), dcc.Graph(id="animation-graph-plate2")],
        overlay_style={"visibility": "visible", "opacity": .1, "backgroundColor": "white"},
        type="circle"
    ),
    dcc.Interval(
        id='interval-component',
        interval=16,  # Interval in milliseconds
        n_intervals=0,
        disabled=True
    ),
])

# Define the callback to update the graphs
@app.callback(
    [Output("animation-graph-plate1", "figure"),
     Output("animation-graph-plate2", "figure")],
    [Input("interval-component", "n_intervals")]
)
def update_graph(n_intervals):
    colors = {'Fx1': 'rgba(255, 0, 0, 0.2)', 'Fy1': 'rgba(0, 255, 0, 0.2)', 'Fz1': 'rgba(0, 0, 255, 0.2)',
              'Fx2': 'rgba(255, 165, 0, 0.2)', 'Fy2': 'rgba(0, 255, 255, 0.2)', 'Fz2': 'rgba(128, 0, 128, 0.2)'}

    frames1 = [
        go.Frame(
            data=[
                go.Scatter(
                    x=df["time (s)"][:(k + 1) * (int(len(df["time (s)"]) / TotalFrames))],
                    y=df["Fx1"][:(k + 1) * (int(len(df["Fx1"]) / TotalFrames))],
                    mode='lines',
                    fill='tozeroy',
                    fillcolor=colors["Fx1"],
                    name='Fx1'
                ),
                go.Scatter(
                    x=df["time (s)"][:(k + 1) * (int(len(df["time (s)"]) / TotalFrames))],
                    y=df["Fy1"][:(k + 1) * (int(len(df["Fy1"]) / TotalFrames))],
                    mode='lines',
                    fill='tozeroy',
                    fillcolor=colors["Fy1"],
                    name='Fy1'
                ),
                go.Scatter(
                    x=df["time (s)"][:(k + 1) * (int(len(df["time (s)"]) / TotalFrames))],
                    y=df["Fz1"][:(k + 1) * (int(len(df["Fz1"]) / TotalFrames))],
                    mode='lines',
                    fill='tozeroy',
                    fillcolor=colors["Fz1"],
                    name='Fz1'
                )
            ],
            name=str(k)
        ) for k in range(TotalFrames)
    ]

    frames2 = [
        go.Frame(
            data=[
                go.Scatter(
                    x=df["time (s)"][:(k + 1) * (int(len(df["time (s)"]) / TotalFrames))],
                    y=df["Fx2"][:(k + 1) * (int(len(df["Fx2"]) / TotalFrames))],
                    mode='lines',
                    fill='tozeroy',
                    fillcolor=colors["Fx2"],
                    name='Fx2'
                ),
                go.Scatter(
                    x=df["time (s)"][:(k + 1) * (int(len(df["time (s)"]) / TotalFrames))],
                    y=df["Fy2"][:(k + 1) * (int(len(df["Fy2"]) / TotalFrames))],
                    mode='lines',
                    fill='tozeroy',
                    fillcolor=colors["Fy2"],
                    name='Fy2'
                ),
                go.Scatter(
                    x=df["time (s)"][:(k + 1) * (int(len(df["time (s)"]) / TotalFrames))],
                    y=df["Fz2"][:(k + 1) * (int(len(df["Fz2"]) / TotalFrames))],
                    mode='lines',
                    fill='tozeroy',
                    fillcolor=colors["Fz2"],
                    name='Fz2'
                )
            ],
            name=str(k)
        ) for k in range(TotalFrames)
    ]

    figure1 = go.Figure(
        data=[
            go.Scatter(
                x=df["time (s)"],
                y=df["Fx1"],
                mode='lines',
                fill='tozeroy',
                fillcolor=colors["Fx1"],
                name='Fx1'
            ),
            go.Scatter(
                x=df["time (s)"],
                y=df["Fy1"],
                mode='lines',
                fill='tozeroy',
                fillcolor=colors["Fy1"],
                name='Fy1'
            ),
            go.Scatter(
                x=df["time (s)"],
                y=df["Fz1"],
                mode='lines',
                fill='tozeroy',
                fillcolor=colors["Fz1"],
                name='Fz1'
            )
        ],
        layout=go.Layout(
            title='Force Plate 1: Force Components vs Time',
            xaxis=dict(title='Time (s)', range=[df["time (s)"].min(), df["time (s)"].max()]),
            yaxis=dict(title='Force (N)', range=[min(df["Fx1"].min(), df["Fy1"].min(), df["Fz1"].min()), max(df["Fx1"].max(), df["Fy1"].max(), df["Fz1"].max())]),
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 16, "redraw": False},
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
                        "label": f"{df['time (s)'][k * int(len(df) / TotalFrames)]:.3f}",
                        "method": "animate"
                    } for k in range(TotalFrames)
                ]
            }]
        ),
        frames=frames1
    )

    figure2 = go.Figure(
        data=[
            go.Scatter(
                x=df["time (s)"],
                y=df["Fx2"],
                mode='lines',
                fill='tozeroy',
                fillcolor=colors["Fx2"],
                name='Fx2'
            ),
            go.Scatter(
                x=df["time (s)"],
                y=df["Fy2"],
                mode='lines',
                fill='tozeroy',
                fillcolor=colors["Fy2"],
                name='Fy2'
            ),
            go.Scatter(
                x=df["time (s)"],
                y=df["Fz2"],
                mode='lines',
                fill='tozeroy',
                fillcolor=colors["Fz2"],
                name='Fz2'
            )
        ],
        layout=go.Layout(
            title='Force Plate 2: Force Components vs Time',
            xaxis=dict(title='Time (s)', range=[df["time (s)"].min(), df["time (s)"].max()]),
            yaxis=dict(title='Force (N)', range=[min(df["Fx2"].min(), df["Fy2"].min(), df["Fz2"].min()), max(df["Fx2"].max(), df["Fy2"].max(), df["Fz2"].max())]),
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 16, "redraw": False},
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
                        "label": f"{df['time (s)'][k * int(len(df) / TotalFrames)]:.3f}",
                        "method": "animate"
                    } for k in range(TotalFrames)
                ]
            }]
        ),
        frames=frames2
    )

    return figure1, figure2

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
