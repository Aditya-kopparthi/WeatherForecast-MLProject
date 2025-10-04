from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load your trained model
model = pickle.load(open("weather_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Collect form inputs
    features = [float(x) for x in request.form.values()]
    final_input = np.array([features])  # Convert to 2D array
    prediction = model.predict(final_input)[0]

    return render_template("index.html",
                           prediction_text=f"🌤️ Predicted Weather: {prediction}")

if __name__ == "__main__":
    app.run(debug=True)
