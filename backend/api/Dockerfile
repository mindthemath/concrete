FROM python:3.13-slim
WORKDIR /app
COPY backend/api/requirements.txt backend/api/server.py .
COPY frontend/data ./data
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "server.py"]
