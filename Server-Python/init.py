from flask import Flask, request, jsonify
import json
from PIL import Image
import io

app = Flask(__name__)

@app.route("/face-depth-data", methods = ['POST'])
def face_data():
    if 'depthData' not in request.form or 'image' not in request.files:
        return jsonify({"error": "Falta datos"}), 400
    
    depth_data = json.load(request.form["depthData"])
    
    print(request.form)
    
    with open("./Data/FaceDepth.json", "w") as outfile:
        json.dump(depth_data, outfile)

    print("Datos de profundidad", depth_data)

    image_file = request.files["image"]
    image = Image.open(io.BytesIO(image_file.read()))
    image.save("./Data/FaceImage.jpg")

    return jsonify({"Message": "Datos recibidos con exito"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)