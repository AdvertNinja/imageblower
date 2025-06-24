import os
import sys
import requests
import torch
import numpy as np
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from io import BytesIO
from PIL import Image

# Import modelových tříd
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

app = Flask(__name__)
CORS(app, resources={r"/upscale": {"origins": "https://cemex.advert.ninja"}})

model = None

@app.before_first_request
def load_model():
    global model
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = os.path.join("weights", "realesr-general-x4v3.pth")

        # Stáhni model, pokud neexistuje
        if not os.path.exists(model_path):
            print("[MODEL] Stahuji model z URL...")
            url = "https://cemex.advert.ninja/tools/imageblower/weights/realesr-general-x4v3.pth"
            os.makedirs("weights", exist_ok=True)
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(model_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print("[MODEL] Model úspěšně stažen.")

        # Vytvoření instance RRDBNet
        model_net = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=66,         # 💡 doporučeno pro realesr-general-x4v3
            num_grow_ch=32,
            scale=4
        )

        model = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model_net,     # ✅ hlavní oprava zde
            device=device,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False
        )

        print("[MODEL] Model načten a připraven.")

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
    input_np = np.array(image)[:, :, ::-1]  # RGB → BGR

    try:
        output_np, _ = model.enhance(input_np, outscale=1)
        output_img = Image.fromarray(output_np[:, :, ::-1])  # BGR → RGB

        output = BytesIO()
        output_img.save(output, format='PNG')
        output.seek(0)
        return send_file(output, mimetype='image/png')

    except Exception as e:
        print(f"[ERROR] Upscaling selhal: {e}")
        return jsonify({'error': 'Upscaling failed', 'message': str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
