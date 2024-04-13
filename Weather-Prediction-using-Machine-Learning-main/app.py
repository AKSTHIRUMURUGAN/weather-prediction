# Define means and standard deviations for each feature
mean_temp = 25.0
mean_press = 101.0
mean_rel_hum = 50.0
mean_wind_speed = 10.0
mean_visibility = 10.0
mean_hour = 12.0

std_temp = 5.0
std_press = 5.0
std_rel_hum = 10.0
std_wind_speed = 5.0
std_visibility = 5.0
std_hour = 6.0

from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the model and label encoder
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Extract features from form data
        temp = float(request.form['temperature_C'])
        press_kpa = float(request.form['pressure_kpa'])
        rel_hum = float(request.form['relative_humidity'])
        wind_speed = float(request.form['wind_speed_kmph'])
        visibility_km = float(request.form['visibility_km'])
        hour = float(request.form['hour'])

        # Manually scale the input data (standardization)
        means = [mean_temp, mean_press, mean_rel_hum, mean_wind_speed, mean_visibility, mean_hour]
        stds = [std_temp, std_press, std_rel_hum, std_wind_speed, std_visibility, std_hour]

        scaled_temp = (temp - means[0]) / stds[0]
        scaled_press = (press_kpa - means[1]) / stds[1]
        scaled_rel_hum = (rel_hum - means[2]) / stds[2]
        scaled_wind_speed = (wind_speed - means[3]) / stds[3]
        scaled_visibility_km = (visibility_km - means[4]) / stds[4]
        scaled_hour = (hour - means[5]) / stds[5]

        # Prepare the input data for prediction
        scaled_data = [[scaled_temp, scaled_press, scaled_rel_hum, scaled_wind_speed, scaled_visibility_km, scaled_hour]]

        # Make prediction
        pred_label_index = model.predict(scaled_data)[0]
        pred_label = label_encoder.inverse_transform([pred_label_index])[0]

        return render_template('index.html', prediction_text=f"The current weather is {pred_label.lower()}.")

    # If method is GET, render the form
    return render_template('index.html')

if __name__ == "__main__":
    app.run(port=8080)
