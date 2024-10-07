from flask import Flask, request, jsonify
import json
from PIL import Image
import io
import os

app = Flask(__name__)

@app.route("/face-depth-data", methods=["POST"])
def face_data():
    if "faceImage" not in request.files or "depthImage" not in request.files:
        return jsonify({"error": "Falta datos"}), 400

    depth_file = request.files["depthImage"]
    depth_image = Image.open(io.BytesIO(depth_file.read()))
    depth_image = depth_image.rotate(270)
    depth_image.save(os.path.join(os.path.dirname(__file__), "data", "DepthImage.jpg"))

    colorImage_file = request.files["faceImage"]
    colorImage = Image.open(io.BytesIO(colorImage_file.read()))
    colorImage = colorImage.rotate(270)
    colorImage.save(os.path.join(os.path.dirname(__file__), "data", "FaceImage.jpg"))

    return jsonify({"Message": "Datos recibidos con exito"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
