from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np, os

app = Flask(__name__)
UPLOAD = "static/uploads"
os.makedirs(UPLOAD, exist_ok=True)

model = load_model("mask_detector.h5")

def predict(path):
    img = image.load_img(path, target_size=(150,150))
    img = np.expand_dims(image.img_to_array(img)/255, 0)
    return "Mask ON 😷" if model.predict(img)[0][0] < 0.5 else "No Mask"

@app.route("/", methods=["GET","POST"])
def home():
    if request.method=="POST":
        f = request.files["file"]
        path = os.path.join(UPLOAD, f.filename)
        f.save(path)
        return render_template("index.html", img_path=path, result=predict(path))
    return render_template("index.html")

app.run(debug=True)
