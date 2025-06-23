from flask import Flask, request, send_file
from PIL import Image
import io
import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

app = Flask(__name__)

# Inicializace modelu při startu
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
upsampler = RealESRGANer(
    scale=2,
    model_path='weights/RealESRGAN_x2plus.pth',
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False,
    device=torch.device('cpu')
)

@app.route('/upscale', methods=['POST'])
def upscale():
    if 'image' not in request.files:
        return {'success': False, 'error': 'No image uploaded'}, 400

    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    output, _ = upsampler.enhance(img)

    buf = io.BytesIO()
    output.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')
