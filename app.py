from flask import Flask, render_template, request, flash
import pickle
import numpy as np
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Needed for flashing messages

# Load model
MODEL_PATH = 'heart_failure_rf.pkl'
if not os.path.exists(MODEL_PATH):
    raise Exception('Model file not found. Please make sure "heart_failure_rf.pkl" exists.')

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

feature_names = [
    "age","anaemia","creatinine_phosphokinase","diabetes","ejection_fraction",
    "high_blood_pressure","platelets","serum_creatinine","serum_sodium",
    "sex","smoking","time"
]

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    probability = None
    error_msg = None

    if request.method == 'POST':
        try:
            features = []
            for name in feature_names:
                value = request.form.get(name)
                if value is None or value == '':
                    error_msg = f"Value for {name} is required."
                    break
                if name in ['platelets', 'serum_creatinine', 'age']:
                    features.append(float(value))
                else:
                    features.append(int(value))
            if error_msg is None:
                arr = np.array([features])
                pred = model.predict(arr)[0]
                probability = model.predict_proba(arr)[0][1]  # Probability of death event
                result = "High Risk: Patient likely to DIE" if pred == 1 else "Low Risk: Patient likely to SURVIVE"
        except Exception as e:
            error_msg = f"Error processing input: {e}"
    return render_template(
        'index.html', 
        result=result,
        probability=probability,
        feature_names=feature_names,
        error_msg=error_msg
    )

if __name__ == '__main__':
    app.run(debug=True)
