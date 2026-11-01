import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import generativeai as genai

from model_runtime import StudentSuccessModel

app = Flask(__name__)
allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]
CORS(app, resources={r"/api/*": {"origins": allowed_origins}})

api_key = os.environ.get("GEMINI_API_KEY", "")
if api_key:
    genai.configure(api_key=api_key)
client = genai.GenerativeModel("gemini-2.0-flash") if api_key else None
trained_model = StudentSuccessModel()


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "genai_configured": bool(api_key),
        "trained_model_available": trained_model.configured,
    })


@app.post("/api/predict")
def predict_demo():
    """Lightweight demo scorer retained for the current three-field frontend."""
    data = request.get_json(silent=True) or {}
    try:
        gpa = float(data["gpa"])
        attendance = float(data["attendance"])
        study_hours = float(data["studyHours"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "gpa, attendance and studyHours must be numeric"}), 400

    if not 0 <= gpa <= 4:
        return jsonify({"error": "GPA must be between 0 and 4"}), 400
    if not 0 <= attendance <= 100:
        return jsonify({"error": "Attendance must be between 0 and 100"}), 400
    if not 0 <= study_hours <= 168:
        return jsonify({"error": "Study hours must be between 0 and 168"}), 400

    score = (gpa / 4.0) * 0.4 + (attendance / 100.0) * 0.3 + min(study_hours / 20.0, 1.0) * 0.3
    probability = min(max(score, 0.1), 0.95)
    label = "Likely to Succeed" if probability > 0.5 else "At Risk"
    confidence = int((probability if probability > 0.5 else 1 - probability) * 100)
    return jsonify({"prediction": label, "confidence": confidence, "mode": "demo"})


@app.post("/api/predict-trained")
def predict_trained():
    """Run the actual three-class TensorFlow model when local artifacts are present."""
    if not trained_model.configured:
        return jsonify({"error": "Trained model artifacts are not available locally"}), 503
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Expected a JSON object containing enrollment features"}), 400
    try:
        result = trained_model.predict(data)
        return jsonify({**result, "mode": "trained_model"})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        return jsonify({"error": "Unable to run trained model prediction"}), 500


def _bounded_text(value, *, maximum=4000):
    text = str(value or "").strip()
    if not text:
        return ""
    return text[:maximum]


@app.post("/api/genai/summary")
def genai_summary():
    if not client:
        return jsonify({"error": "GEMINI_API_KEY is not configured"}), 503
    data = request.get_json(silent=True) or {}
    context = _bounded_text(data.get("context"))
    if not context:
        return jsonify({"error": "Missing required field: context"}), 400

    try:
        prompt = f"Provide concise, actionable feedback about the student based on this context: {context}"
        response = client.generate_content(prompt)
        return jsonify({"summary": response.text})
    except Exception:
        return jsonify({"error": "Unable to generate summary"}), 500


@app.post("/api/chat")
def chat():
    if not client:
        return jsonify({"error": "GEMINI_API_KEY is not configured"}), 503
    data = request.get_json(silent=True) or {}
    messages = data.get("messages")
    if not isinstance(messages, list) or not messages or len(messages) > 20:
        return jsonify({"error": "messages must contain between 1 and 20 items"}), 400

    contents = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or "role" not in message or "content" not in message:
            return jsonify({"error": "Each message needs role and content"}), 400
        role = "model" if str(message["role"]).lower() == "assistant" else "user"
        content = _bounded_text(message["content"], maximum=2000)
        if not content:
            return jsonify({"error": "Message content cannot be empty"}), 400
        if role == "user" and index == len(messages) - 1:
            content = f"[Keep the response concise and actionable.]\n\n{content}"
        contents.append({"role": role, "parts": [{"text": content}]})

    if contents[-1]["role"] != "user":
        return jsonify({"error": "The last message must be from the user"}), 400

    try:
        response = client.generate_content(
            contents,
            generation_config={"max_output_tokens": 500, "temperature": 0.7},
        )
        return jsonify({"reply": response.text})
    except Exception:
        return jsonify({"error": "Unable to generate chat response"}), 500


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG", "0") == "1", port=8000)
