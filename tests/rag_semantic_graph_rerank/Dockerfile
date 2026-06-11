FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir "sentence-transformers>=3.0"
COPY runner.py /app/runner.py

CMD ["python", "/app/runner.py"]
