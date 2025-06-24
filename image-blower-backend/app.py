import os
import sys
import requests
import torch
import numpy as np
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from io import BytesIO
from PIL import Image

# Import RealESRGAN
from realesrgan import RealESRGAN

app = Flask(__name__)
CORS(app, origins=["https://cemex.advert.ninja"])

model = None

@app.before_first_request
def load_model():
    global model
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = os.path.join("weights", "realesr-general-x4v3.pth")

        # Stáhnout model pokud neexistuje
        if not os.path.exists(model_path):
            print("[MODEL] Stahuji model z externího URL...")
            url = "https://cemex.advert.ninja/tools/imageblower/weights/realesr-general-x4v3.pth"
            os.makedirs("weights", exist_ok=True)
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(model_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print("[MODEL] Model úspěšně stažen.")

        model = RealESRGAN(device, scale=4)
        model.load_weights(model_path)
        print("[MODEL] Model načten.")
    except Exception as e:
        print(f"[MODEL] Chyba při načítání modelu: {e}")

@app.route('/')
def home():
    return 'ImageBlower backend je online 🎈'

@app.route('/ping', methods=['GET'])
def ping():
    return "OK", 200

@app.route('/upscale', methods=['POST'])
def upscale_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image = Image.open(file.stream).convert("RGB")

    try:
        input_np = np.array(image)
        output_np = model.predict(input_np)
        output_img = Image.fromarray(output_np)

        output = BytesIO()
        output_img.save(output, format='PNG')
        output.seek(0)
        return send_file(output, mimetype='image/png')

    except Exception as e:
        return jsonify({'error': 'Upscaling failed', 'message': str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
