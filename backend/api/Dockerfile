FROM python:3.13
WORKDIR /app
COPY data requirements.txt server.py .
RUN pip install -r requirements.txt
CMD ["python", "server.py"]
