from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from io import BytesIO
from PIL import Image
import numpy as np
import os
import sys
import urllib.request
import torch

# Přidej cestu k modulu Real-ESRGAN
sys.path.append(os.path.join(os.path.dirname(__file__), 'real_esrgan_local'))  # přizpůsob název složky

from realesrgan.real_esrgan import RealESRGAN

app = Flask(__name__)
CORS(app, origins=["https://cemex.advert.ninja"])

model = None

@app.before_first_request
def load_model():
    global model
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        weights_dir = "weights"
        os.makedirs(weights_dir, exist_ok=True)

        model_path = os.path.join(weights_dir, "realesr-general-x4v3.pth")
        model_url = "https://cemex.advert.ninja/tools/imageblower/weights/realesr-general-x4v3.pth"

        if not os.path.exists(model_path):
            print(f"[MODEL] Stahuji model z {model_url}...")
            urllib.request.urlretrieve(model_url, model_path)
            print("[MODEL] Model stažen.")

        model = RealESRGAN(device, scale=4)
        model.load_weights(model_path)
        print("[MODEL] Model RealESRGAN načten.")

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
