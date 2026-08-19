FROM python:3.11-slim

WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

# Forma shell (sin corchetes) para que se expanda ${PORT}: Render asigna el
# puerto por variable de entorno y el servicio tiene que escuchar ahí, o el
# despliegue queda sin responder. El 7860 es el valor por defecto para correr
# en local.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}
