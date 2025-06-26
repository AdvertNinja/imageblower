from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import os
import requests
import torch
import numpy as np
from io import BytesIO
from PIL import Image
import gc

from realesrgan import RealESRGANer
from basicsr.archs.srvgg_arch import SRVGGNetCompact

app = Flask(__name__)

# ===== CORS =====
ALLOWED_ORIGINS = [
    "https://cemex.advert.ninja",
    "http://cemex.advert.ninja",
    "http://13.60.168.56:5000",
    "http://localhost:5000"
]
CORS(app, resources={r"/": {"origins": ALLOWED_ORIGINS}})

# ===== Model loader =====
model = None

def load_model():
    global model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join("weights", "realesr-general-x4v3.pth")

    if not os.path.exists(model_path):
        url = "https://cemex.advert.ninja/tools/imageblower/weights/realesr-general-x4v3.pth"
        os.makedirs("weights", exist_ok=True)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(model_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        del r
        gc.collect()

    model_net = SRVGGNetCompact(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_conv=32,
        upscale=4,
        act_type="prelu"
    )

    model = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model_net,
        device=device,
        tile=16,         # 🔻 z 128 na 32
        tile_pad=10,
        pre_pad=0,
        half=False       # můžeš zkusit True, pokud máš GPU
    )

@app.before_first_request
def init_model():
    load_model()

# ===== Root endpoint =====
@app.route("/", methods=["GET", "POST", "OPTIONS"])
def root():
    if request.method == "OPTIONS":
        return "", 200

    if request.method == "GET":
        return "ImageBlower backend je online 🎈", 200

    if "image" not in request.files:
        return jsonify(error="No image uploaded"), 400

    file = request.files["image"]
    image = Image.open(file.stream).convert("RGB")
    input_np = np.array(image)[:, :, ::-1]  # RGB → BGR

    try:
        output_np, _ = model.enhance(input_np, outscale=2)
        output_img = Image.fromarray(output_np[:, :, ::-1])  # BGR → RGB

        buf = BytesIO()
        output_img.save(buf, format="PNG")
        buf.seek(0)

        # cleanup
        del input_np, output_np, output_img
        torch.cuda.empty_cache()
        gc.collect()

        return send_file(buf, mimetype="image/png")
    except Exception as e:
        return jsonify(error="Upscaling failed", message=str(e)), 500

# ===== Simple ping =====
@app.route("/ping", methods=["GET"])
def ping():
    return "OK", 200

# ===== Run locally =====
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
