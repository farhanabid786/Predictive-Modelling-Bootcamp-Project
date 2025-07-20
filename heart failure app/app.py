from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get input values from form
    features = [float(request.form[key]) for key in request.form]
    
    # Scale input
    final_input = scaler.transform([features])
    
    # Predict
    prediction = model.predict(final_input)[0]
    
    result = "High Risk of Heart Failure 😟" if prediction == 1 else "Low Risk (Safe) 😊"
    
    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
