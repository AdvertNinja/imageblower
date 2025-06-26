# Použijeme oficiální Python image jako základ
FROM python:3.10-slim

# Nainstalujeme potřebné systémové knihovny
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Nastavíme pracovní adresář v kontejneru
WORKDIR /app

# Zkopírujeme requirements a nainstalujeme Python závislosti
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Zkopírujeme zbytek aplikace
COPY . .

# Nastavíme spouštěcí příkaz
CMD ["python", "image-blower-backend/app.py"]
