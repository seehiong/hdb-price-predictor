# 🏠 HDB Price Predictor App

A Streamlit-based frontend that predicts Singapore HDB resale flat prices using an XGBoost model served via KServe on Oracle Kubernetes Engine (OKE). This app connects to a model inference endpoint and applies pre-saved data scaling for accurate predictions.

---

## ✨ Features

- Interactive Streamlit UI for user-friendly input  
- Scaler and preprocessing with `joblib`  
- Dockerized and deployable to Kubernetes (e.g. OKE)

---

## 📁 Project Structure

```bash
.
├── app.py # Streamlit app code
├── scaler.joblib # Pre-trained StandardScaler
├── Dockerfile # Container build file
├── requirements.txt # Python dependencies
└── README.md
```

---

## 🚀 Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/seehiong/hdb-price-predictor-app.git
cd hdb-price-predictor-app
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

## 🐳 Docker Instructions

### Build and Run Locally

```bash
docker build -t hdb-predictor-app:latest .
docker run -p 8501:8501 hdb-predictor-app:latest
```

### Tag and Push to Docker Hub

```bash
docker tag hdb-predictor-app:latest <dockerhub-username>/hdb-predictor-app:latest
docker push <dockerhub-username>/hdb-predictor-app:latest
```

## 🌐 Deployment (Kubernetes)

This app is designed to work with KServe and Istio in a Kubernetes environment. Example deployment manifest:

```
env:
- name: KSERVE_URL
  value: "http://<kserve-inference-url>/v2/models/hdb-resale-xgb/infer"
- name: KSERVE_HOST
  value: "hdb-resale-xgb-kserve-test.example.com"
```

## 📄 Related Blog Post

👉 [Deploy KServe on OKE](https://seehiong.github.io/2025/deploy-kserve-on-oke/)


## 🛠 Built With

- Streamlit
- scikit-learn
- KServe
- Oracle Kubernetes Engine (OKE)