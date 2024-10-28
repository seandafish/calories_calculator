from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

pipeline = joblib.load('calories_burned_model.pkl')

def predict_calories(age, gender, weight, height, avg_bpm, session_duration, bmi, workout_type):
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Weight (kg)': [weight],
        'Height (m)': [height],
        'Avg_BPM': [avg_bpm],
        'Session_Duration (hours)': [session_duration],
        'BMI': [bmi],
        'Workout_Type': [workout_type]
    })
    return pipeline.predict(input_data)[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    gender = request.form['gender']
    weight = float(request.form['weight'])
    height = float(request.form['height'])
    avg_bpm = float(request.form['avg_bpm'])
    session_duration = float(request.form['session_duration'])
    bmi = float(request.form['bmi'])
    workout_type = request.form['workout_type']
    
    # Make the prediction
    predicted_calories = predict_calories(age, gender, weight, height, avg_bpm, session_duration, bmi, workout_type)
    
    # Return the result
    return render_template('index.html', result=f"Estimated Calories Burned: {predicted_calories:.2f}")

if __name__ == '__main__':
    app.run(debug=True)
