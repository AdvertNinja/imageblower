import os
import subprocess
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from io import BytesIO
from PIL import Image

# 🟢 Inicializace modelu
model_path = 'weights/RealESRGAN_x2plus.pth'
os.makedirs('weights', exist_ok=True)

if not os.path.exists(model_path):
    print("[INIT] Stahuji model...")
    result = subprocess.run([
        "curl", "-L", "-o", model_path,
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5/RealESRGAN_x2plus.pth"
    ])
    if result.returncode != 0:
        print("[ERROR] Selhalo stažení modelu.")
        exit(1)

# 🟢 Flask server
app = Flask(__name__)
CORS(app, origins=["https://cemex.advert.ninja"])

@app.route('/')
def home():
    return 'ImageBlower backend je online 🎈'

@app.route('/ping', methods=['GET'])
def ping():
    print("[PING] Server byl pingnut.")
    return "OK", 200

@app.route('/upscale', methods=['POST'])
def upscale_image():
    if 'image' not in request.files:
        print("[SERVER] Obrázek nebyl nahrán.")
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    print(f"[SERVER] Přijatý soubor: {file.filename}")

    try:
        image = Image.open(file.stream)
        new_size = (image.width * 2, image.height * 2)
        upscaled = image.resize(new_size, Image.LANCZOS)
        print(f"[SERVER] Upscalováno z {image.size} na {new_size}")

        output = BytesIO()
        upscaled.save(output, format='PNG')
        output.seek(0)
        return send_file(output, mimetype='image/png')

    except Exception as e:
        print(f"[SERVER] Chyba při zpracování obrázku: {e}")
        return jsonify({'error': 'Upscaling failed', 'message': str(e)}), 500

if __name__ == '__main__':
    print("[SERVER] Spouštím backend na 0.0.0.0:10000")
    app.run(host='0.0.0.0', port=10000)
