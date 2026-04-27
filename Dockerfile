FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y default-jdk && rm -rf /var/lib/apt/lists/*
COPY . .
ENV PYTHONUNBUFFERED=1
ENV PYSPARK_PYTHON=python3
CMD ["python", "Inference.py"]