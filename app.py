from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import csv
from datetime import datetime

app = Flask(__name__)

model = load_model("animal_model.keras")

with open("labels.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

os.makedirs("static", exist_ok=True)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['image']

    filepath = "static/" + file.filename
    file.save(filepath)

    img = image.load_img(filepath, target_size=(224,224))
    img = image.img_to_array(img)

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    print("Raw prediction:", prediction)

    predicted_class = np.argmax(prediction)

    result = classes[predicted_class]
    confidence = np.max(prediction) * 100

    with open("history.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            filepath,
            result,
            round(confidence, 2)
        ])

    return render_template(
        "index.html",
        prediction=result,
        confidence=round(confidence, 2),
        image=filepath
    )


@app.route('/history')
def history():

    data = []

    if os.path.exists("history.csv"):
        with open("history.csv", "r") as f:
            reader = csv.reader(f)
            data = list(reader)

    return render_template("history.html", data=data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)