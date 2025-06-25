# Dockerfile
FROM python:3.10-slim

# Nastaví pracovní složku v kontejneru
WORKDIR /app

# Zkopíruje requirements a nainstaluje závislosti
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Zkopíruje zbytek kódu
COPY . .

# Spustí aplikaci
CMD ["python", "image-blower-backend/app.py"]
