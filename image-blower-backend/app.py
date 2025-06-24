from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from io import BytesIO
from PIL import Image
import time

# Inicializace Flask app
app = Flask(__name__)
CORS(app, origins=["https://cemex.advert.ninja"])  # povolí požadavky z tvého webu

@app.route('/')
def home():
    return 'ImageBlower backend je online 🎈'

@app.route('/upscale', methods=['POST'])
def upscale_image():
    if 'image' not in request.files:
        print("[SERVER] Obrázek nebyl nahrán.")
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    print(f"[SERVER] Přijatý soubor: {file.filename}")

    try:
        image = Image.open(file.stream)
        original_size = image.size

        # Vytvoř jednoduchou "upscale" logiku (např. 2× větší rozměr)
        upscale_factor = 2
        new_size = (image.width * upscale_factor, image.height * upscale_factor)
        upscaled = image.resize(new_size, Image.LANCZOS)

        print(f"[SERVER] Upscalováno z {original_size} na {new_size}")

        # Výstup do paměti
        output = BytesIO()
        upscaled.save(output, format='PNG')
        output.seek(0)

        return send_file(output, mimetype='image/png')

    except Exception as e:
        print(f"[SERVER] Chyba při zpracování obrázku: {e}")
        return jsonify({'error': 'Upscaling failed', 'message': str(e)}), 500

# Spuštění serveru
if __name__ == '__main__':
    print("[SERVER] Spouštím backend na 0.0.0.0:10000")
    app.run(host='0.0.0.0', port=10000)
