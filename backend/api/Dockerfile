FROM python:3.13
WORKDIR /app
COPY backend/api/requirements.txt backend/api/server.py .
COPY frontend/data ./data
RUN pip install -r requirements.txt
CMD ["python", "server.py"]
