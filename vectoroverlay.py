from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import json
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'video' not in request.files:
            return redirect(request.url)

        video_file = request.files['video']
        if video_file.filename == '':
            return redirect(request.url)

        if video_file:
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
            video_file.save(video_path)

            # Read the .csv file
            data = pd.read_csv('data/SM_SbS_02_Raw_Data - SM_SoftballSwing_Trial2_Raw_Data.csv')

            # Process the data: Extract specific columns and rows
            x_coords = data.iloc[18:10000, 6].astype(float).tolist()
            y_coords = data.iloc[18:10000, 5].astype(float).tolist()
            force_x = data.iloc[18:10000, 2].astype(float).tolist()
            force_y = data.iloc[18:10000, 1].astype(float).tolist()

            # Save the data to a JSON file for the HTML to use
            output_data = {
                'x_coords': x_coords,
                'y_coords': y_coords,
                'force_x': force_x,
                'force_y': force_y
            }

            json_path = os.path.join(app.static_folder, 'data.json')
            with open(json_path, 'w') as outfile:
                json.dump(output_data, outfile)

            return redirect(url_for('index', video_filename=video_file.filename))

    video_filename = request.args.get('video_filename', None)
    return render_template('vectoroverlay.html', video_filename=video_filename)

if __name__ == '__main__':
    app.run(debug=True)
