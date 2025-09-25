# from flask import Flask, render_template, request
# import pickle
# import pandas as pd
# import os

# app = Flask(__name__, template_folder=os.path.join(os.getcwd(), "templates"))

# @app.route('/')
# def home():
#     print("Rendering index.html")
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         batting_team = request.form['batting_team']
#         bowling_team = request.form['bowling_team']
#         selected_city = request.form['selected_city']
#         target = int(request.form['target'])
#         score = int(request.form['score'])
#         balls_left = int(request.form['balls_left'])
#         wickets = int(request.form['wickets'])

#         runs_left = target - score
#         wickets_remaining = 10 - wickets
#         overs_completed = (120 - balls_left) / 6 if balls_left != 120 else 0.1  # Avoid division by zero
#         crr = score / overs_completed
#         rrr = runs_left / (balls_left / 6)

#         input_data = pd.DataFrame({
#             'batting_team': [batting_team],
#             'bowling_team': [bowling_team],
#             'city': [selected_city],
#             'runs_left': [runs_left],
#             'balls_left': [balls_left],
#             'wickets_remaining': [wickets_remaining],
#             'total_run_x': [target],
#             'crr': [crr],
#             'rrr': [rrr]
#         })

#         model_path = os.path.join("model", "ra_pipe.pkl")
#         if not os.path.exists(model_path):
#             return "Model file not found!", 500

#         with open(model_path, 'rb') as model_file:
#             pipe = pickle.load(model_file)

#         result = pipe.predict_proba(input_data)

#         win_probability = round(result[0][1] * 100)
#         loss_probability = round(result[0][0] * 100)

#         return render_template('result.html', batting_team=batting_team, bowling_team=bowling_team, win_probability=win_probability, loss_probability=loss_probability)

# if __name__ == '__main__':
#     app.run(debug=True) 

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import joblib
import os
venue_to_city = {
    'ACA-VDCA Stadium, Visakhapatnam': 'Visakhapatnam',
    'Barabati Stadium, Cuttack': 'Cuttack',
    'Dr DY Patil Sports Academy, Mumbai': 'Mumbai',
    'Dubai International Cricket Stadium, Dubai': 'Dubai',
    'Eden Gardens, Kolkata': 'Kolkata',
    'Feroz Shah Kotla, Delhi': 'Delhi',
    'Himachal Pradesh Cricket Association Stadium, Dharamshala': 'Dharamshala',
    'Holkar Cricket Stadium, Indore': 'Indore',
    'JSCA International Stadium Complex, Ranchi': 'Ranchi',
    'M Chinnaswamy Stadium, Bangalore': 'Bangalore',
    'MA Chidambaram Stadium, Chepauk': 'Chennai',
    'Maharashtra Cricket Association Stadium, Pune': 'Pune',
    'Punjab Cricket Association Stadium, Mohali': 'Mohali',
    'Raipur International Cricket Stadium, Raipur': 'Raipur',
    'Rajiv Gandhi International Stadium, Uppal': 'Hyderabad',
    'Sardar Patel Stadium, Motera': 'Ahmedabad',
    'Sawai Mansingh Stadium, Jaipur': 'Jaipur',
    'Sharjah Cricket Stadium, Sharjah': 'Sharjah',
    'Sheikh Zayed Stadium, Abu-Dhabi': 'Abu Dhabi',
    'Wankhede Stadium, Mumbai': 'Mumbai'
}

app = Flask(__name__)

# Load models
ra_pipe = pickle.load(open("model/ra_pipe.pkl", "rb"))        # win probability model
pipe = pickle.load(open("model/pipe.pkl", "rb"))              # ML-based score prediction model
ridge_model = joblib.load("iplmodel_ridge.sav")         # Ridge regression model
scaler = joblib.load("model/scaler.pkl")                      # Scaler for ridge input

@app.route('/')
def home():
    return render_template('index.html') 

@app.route('/predict', methods=['POST'])
def predict():
    # Shared input
    batting_team = request.form['batting_team']
    bowling_team = request.form['bowling_team']
    # selected_city = request.form['venue']
    selected_venue = request.form['venue']
    selected_city = venue_to_city.get(selected_venue, 'Unknown')
    target = int(request.form['target'])
    score = int(request.form['score'])
    balls_left = int(request.form['balls_left'])
    wickets = int(request.form['wickets'])

    runs_left = target - score
    wickets_remaining = 10 - wickets
    overs_completed = (120 - balls_left) / 6 if balls_left != 120 else 0.1
    crr = score / overs_completed
    rrr = runs_left / (balls_left / 6)

    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets_remaining': [wickets_remaining],
        'total_run_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    # Map full venue names to cities used during model training
    # Win probability prediction
    win_probs = ra_pipe.predict_proba(input_df)
    win_percent = round(win_probs[0][1] * 100)
    loss_percent = round(win_probs[0][0] * 100)
    predicted_score_pipe = int(pipe.predict(input_df)[0])

    # Score prediction (Ridge model)
    overs = float(request.form['overs'])
    runs_prev_5 = int(request.form['runs_in_prev_5'])
    wickets_prev_5 = int(request.form['wickets_in_prev_5'])
    b = np.array([[score, wickets, overs_completed, runs_prev_5, wickets_prev_5]])
    b_scaled = scaler.transform(b)
    a = np.zeros((1, 20 + 8 + 8))  # 20 venues, 8 teams x 2
    ridge_input = np.concatenate((a, b_scaled), axis=1)
    ridge_score = int(ridge_model.predict(ridge_input)[0])

    return render_template("result.html",
        batting_team=batting_team,
        bowling_team=bowling_team,
        win_probability=win_percent,
        loss_probability=loss_percent,
        predicted_score_pipe=predicted_score_pipe,
        predicted_score_ridge=f"{ridge_score - 5} to {ridge_score + 10}"
    )

if __name__ == '__main__':
    app.run(debug=True)

