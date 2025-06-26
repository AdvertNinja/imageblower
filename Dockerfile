# ─────────────────────────────────────────────────────────────
# Image: advertninja/imageblower  (Python 3.10 + OpenCV CPU)
# ─────────────────────────────────────────────────────────────
FROM python:3.10-slim

# ▸ Systémové závislosti pro OpenCV (libGL + glib)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ▸ Python závislosti
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ▸ Kód aplikace
COPY . .

# ▸ Spuštění backendu
CMD ["python", "image-blower-backend/app.py"]
