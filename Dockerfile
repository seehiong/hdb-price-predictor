# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY scaler.joblib .
EXPOSE 8501
ENV KSERVE_URL="http://217.142.185.27:80/v2/models/hdb-resale-xgb/infer"
ENV KSERVE_HOST="hdb-resale-xgb-kserve-test.example.com"
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]