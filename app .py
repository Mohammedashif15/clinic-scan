from flask import Flask, request, jsonify, render_template
import os
from model import predict
from utils import draw_boxes

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})

    # Safe paths
    img_path = os.path.join(UPLOAD_FOLDER, file.filename)
    out_path = os.path.join(OUTPUT_FOLDER, file.filename)

    file.save(img_path)

    # Run model prediction
    detections = predict(img_path)

    if len(detections) > 0:
        draw_boxes(img_path, detections, out_path)
        status="Abnormal"
        output_image = f"/static/outputs/{file.filename}"
    else:
        status= "Normal"
        output_image = f"/static/uploads/{file.filename}"

    return jsonify({
        "output": output_image,
        "detections": detections,
        "status": status
    })

if __name__ == "__main__":
    app.run(debug=True)
