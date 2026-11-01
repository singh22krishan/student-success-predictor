# Student Success Predictor

A full-stack academic AI application for exploring student-success risk using enrollment-time data, a TensorFlow training pipeline, a Flask REST API, a React/Vite frontend, and optional Gemini-powered student insights.

## Portfolio Context

This repository is a standalone portfolio version maintained by **Krishan Singh**. The implementation was developed for an academic software/AI project at Centennial College and shared with the course group for submission. The repository has been reorganized for portfolio use, security, testability, and maintainability.

## What It Demonstrates

- End-to-end ML workflow using the UCI **Predict Students' Dropout and Academic Success** dataset
- Leakage-aware feature selection using enrollment-time information only
- Scikit-learn preprocessing with one-hot encoding and standardization
- TensorFlow/Keras multiclass neural-network training
- Artifact-backed inference through the Flask API
- Explicit separation between the lightweight demo scorer and the trained model
- React/Vite UI with prediction, results, GenAI, and metrics views
- Gemini integration through environment configuration
- Restricted configurable CORS, bounded GenAI inputs, generic production errors
- Pytest API coverage + GitHub Actions CI

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

These are recorded experiment results, not marketing/demo numbers.

## Application Architecture

```text
React / Vite frontend
        |
        v
Flask API
  |-- /api/predict          -> lightweight three-field demo scorer
  |-- /api/predict-trained  -> saved TensorFlow/preprocessor artifacts
  |-- /api/genai/summary    -> optional Gemini integration
  |-- /api/chat             -> optional Gemini integration
  +-- /api/health
        |
        v
models/
  |-- student_success_model.keras
  |-- preprocessor.joblib
  |-- label_encoder.joblib
  +-- model_metadata.json

ml/train_model.py -> UCI dataset 697 -> preprocessing -> training -> artifacts
```

## Prediction Modes

### Demo scorer

`POST /api/predict` accepts `gpa`, `attendance`, and `studyHours`. It is intentionally labeled with `"mode": "demo"` and is not presented as the trained TensorFlow model.

### Trained model

`POST /api/predict-trained` loads the artifacts produced by `ml/train_model.py` when they are available locally. The endpoint validates that all enrollment-time features expected by the saved metadata are present, then returns the predicted class, confidence, and per-class probabilities.

If the artifacts have not been generated, the endpoint returns an explicit service-unavailable response instead of silently falling back to demo logic.

## Run Locally

### Backend

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python backend/app.py
```

Optional environment variables:

```text
GEMINI_API_KEY=...
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
FLASK_DEBUG=0
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

The script downloads UCI dataset 697 at runtime and writes trained artifacts to `models/`. Generated model binaries are not committed.

## Tests

```bash
pytest -q tests
```

GitHub Actions runs the Flask API tests on pushes and pull requests.

## Security

- No API keys or trained-model binaries are committed.
- CORS is restricted through `CORS_ORIGINS` rather than opened globally.
- GenAI context/message sizes are bounded.
- Provider/internal exception details are not returned to clients.
- Flask debug mode is opt-in rather than enabled by default.

## Tech Stack

**Python · TensorFlow/Keras · Scikit-learn · Pandas · NumPy · Flask · Pytest · React · Vite · JavaScript · Gemini API · GitHub Actions**
