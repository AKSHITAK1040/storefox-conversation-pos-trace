FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py cabs_engine.py app.py .
COPY README.md .

CMD ["python", "main.py"]
