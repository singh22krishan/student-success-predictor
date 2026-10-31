# Student Success Predictor

A full-stack academic AI application for exploring student-success risk using enrollment-time data, a TensorFlow training pipeline, a Flask REST API, a React/Vite frontend, and optional Gemini-powered student insights.

## Portfolio Context

This repository is a standalone portfolio version maintained by **Krishan Singh**. The implementation was developed for an academic software/AI project at Centennial College and shared with the course group for submission. The repository has been reorganized for portfolio use, security, and maintainability.

## What It Demonstrates

- End-to-end ML workflow using the UCI **Predict Students' Dropout and Academic Success** dataset
- Leakage-aware feature selection using enrollment-time information only
- Scikit-learn preprocessing with one-hot encoding and standardization
- TensorFlow/Keras multiclass neural-network training
- Flask REST API with validation and CORS
- React/Vite UI with prediction, results, GenAI, and metrics views
- Gemini integration using an environment variable rather than hard-coded credentials

## ML Pipeline

The training workflow predicts three outcomes: **Dropout**, **Enrolled**, and **Graduate**.

The recorded notebook run used 4,424 rows from UCI dataset 697, removed semester-performance fields to avoid using future information, and retained 24 enrollment-time features. The data was split into train, validation, and test sets with stratification.

The TensorFlow MLP architecture used:

- Dense 128 + Batch Normalization + Dropout
- Dense 64 + Batch Normalization + Dropout
- Dense 3 + Softmax
- Balanced class weights
- Early stopping and learning-rate reduction

### Recorded Test Results

From the original executed training notebook:

- **Test accuracy:** 58.19%
- **Macro F1:** 54.39%
- **Dropout F1:** 59%
- **Enrolled F1:** 36%
- **Graduate F1:** 68%

These are the recorded experiment results, not marketing/demo numbers.

## Application Architecture

```text
React / Vite frontend
        |
        | REST / JSON
        v
Flask API
  |-- /api/predict
  |-- /api/genai/summary
  |-- /api/chat
  |-- /api/health
        |
        +-- Gemini API (optional, via GEMINI_API_KEY)

ml/train_model.py
        |
        +-- UCI dataset 697
        +-- preprocessing
        +-- TensorFlow model training
        +-- saved artifacts
```

## Important Implementation Note

The current `/api/predict` route is a lightweight demonstration scorer based on GPA, attendance, and study hours. It is intentionally kept separate from the trained three-class TensorFlow pipeline documented in `ml/train_model.py`.

The ML script shows how the production model artifacts are trained and saved. Wiring those artifacts directly into the Flask prediction endpoint is a natural next step.

## Run Locally

### Backend

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"  # optional for GenAI features
python backend/app.py
```

Backend runs on `http://localhost:8000`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The Vite dev server proxies `/api` requests to port 8000.

## Train the Model

```bash
python ml/train_model.py
```

The script downloads the UCI dataset at runtime and writes trained artifacts to `models/`.

## Tech Stack

**Python · TensorFlow/Keras · Scikit-learn · Pandas · NumPy · Flask · REST APIs · React · Vite · JavaScript · Gemini API**

## Security

No API keys are committed. Configure Gemini through the `GEMINI_API_KEY` environment variable.
