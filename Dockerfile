# FuegoBrain — Production Dockerfile
# Base : Python 3.11 slim (image minimale, < 200MB)
FROM python:3.11-slim

# Métadonnées
LABEL maintainer="FuegoDev" \
      project="FuegoBrain" \
      version="1.0.0"

# Variables d'environnement système
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Répertoire de travail
WORKDIR /app

# Installer les dépendances Python (layer cacheable)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY app/ ./app/
COPY demo/ ./demo/

# Exposer le port
EXPOSE 8000

# Commande de démarrage
# --host 0.0.0.0 : nécessaire pour Render (bind sur toutes interfaces)
# --workers 1 : free tier Render = 512MB RAM, 1 worker suffisant
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
