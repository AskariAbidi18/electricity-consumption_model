
# Electricity Consumption Forecasting Model

## 🚀 Project Overview

This project is focused on forecasting electricity consumption using machine learning. It provides a full pipeline to train models and serve predictions via a FastAPI backend and a simple frontend interface.  
There are **two variants available in different branches**:
- ✅ **Daily Model**: Full pipeline including dataset download, model training, FastAPI backend, and frontend for production-ready daily electricity consumption forecasting.
- ⚡ **Minute Model**: Experimental setup for minute-level forecasting with a working pipeline, useful for testing high-frequency prediction ideas.

---

## 🌳 Project Structure

```plaintext
electricity-consumption_model/
├── app/                    # FastAPI backend (app/main.py)
├── frontend/               # Simple frontend (frontend/index.html)
├── notebooks/              # Jupyter notebooks for EDA and experimentation
├── src/                    # Data processing and model training code
├── requirements.txt        # Python dependencies
├── run.py                  # Entry point to download dataset & train model
├── .gitignore
├── README.md
└── ...
```

---

## ⚡ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/AskariAbidi18/electricity-consumption_model.git
cd electricity-consumption_model
```

### 2️⃣ Create and activate a virtual environment

```bash
python -m venv env
source env/bin/activate      # On Windows use: env\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ⚡ Running the Model Pipeline

This will download the dataset, process it, and train the forecasting model:

```bash
python run.py
```

✔ The trained model will be saved locally and ready to be served via API.

---

## ⚡ Running the FastAPI Backend

Once the model is trained, start the API server:

```bash
cd app
uvicorn main:app --reload
```

- The API will be available at `http://127.0.0.1:8000`
- Example endpoint: `http://127.0.0.1:8000/predict`

---

## ⚡ Running the Frontend

After the API server is running:

1. Open `frontend/index.html` in any browser.
2. The frontend will interact with the FastAPI backend to show predictions.

---

## 🌿 Branch Variants

### ✅ Daily Model (Main Branch)

- Full pipeline with dataset download, preprocessing, training, FastAPI backend, and frontend.
- Suitable for practical daily-level electricity forecasting.

### ⚡ Minute Model (Experimental Branch)

- High-frequency (minute-level) model for experimentation.
- Still has the full pipeline and serves predictions, but meant for testing and development.

To switch branches:

```bash
git checkout minute-model
```

---

## 🤝 Contributing

Feel free to fork the repo, create branches, and submit pull requests.

---

## ⚖️ License

MIT License © 2025  
See [LICENSE](LICENSE) for details.

---

> Made with ⚡ by Askari Abidi
