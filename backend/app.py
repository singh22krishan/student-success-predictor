import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import generativeai as genai

app = Flask(__name__)
CORS(app)

api_key = os.environ.get("GEMINI_API_KEY", "")
if api_key:
    genai.configure(api_key=api_key)
client = genai.GenerativeModel("gemini-2.0-flash") if api_key else None


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "genai_configured": bool(api_key)})


@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True) or {}
    gpa = data.get("gpa")
    attendance = data.get("attendance")
    study_hours = data.get("studyHours")

    if None in (gpa, attendance, study_hours):
        return jsonify({"error": "Missing required fields: gpa, attendance, studyHours"}), 400
    if not 0 <= gpa <= 4:
        return jsonify({"error": "GPA must be between 0 and 4"}), 400
    if not 0 <= attendance <= 100:
        return jsonify({"error": "Attendance must be between 0 and 100"}), 400
    if study_hours < 0:
        return jsonify({"error": "Study hours must be non-negative"}), 400

    gpa_score = (gpa / 4.0) * 0.4
    attendance_score = (attendance / 100.0) * 0.3
    study_score = min(study_hours / 20.0, 1.0) * 0.3
    total_score = gpa_score + attendance_score + study_score
    probability = min(max(total_score, 0.1), 0.95)

    if probability > 0.5:
        label = "Likely to Succeed"
        confidence = int(probability * 100)
    else:
        label = "At Risk"
        confidence = int((1 - probability) * 100)

    return jsonify({"prediction": label, "confidence": confidence})


@app.post("/api/genai/summary")
def genai_summary():
    if not client:
        return jsonify({"error": "GEMINI_API_KEY is not configured"}), 503
    data = request.get_json(silent=True) or {}
    context = data.get("context", "").strip()
    if not context:
        return jsonify({"error": "Missing required field: context"}), 400

    try:
        prompt = f"Provide concise, actionable feedback about the student based on this context: {context}"
        response = client.generate_content(prompt)
        return jsonify({"summary": response.text})
    except Exception as exc:
        return jsonify({"error": f"Failed to generate summary: {exc}"}), 500


@app.post("/api/chat")
def chat():
    if not client:
        return jsonify({"error": "GEMINI_API_KEY is not configured"}), 503
    data = request.get_json(silent=True) or {}
    messages = data.get("messages")
    if not isinstance(messages, list) or not messages:
        return jsonify({"error": "messages must be a non-empty list"}), 400

    contents = []
    for index, message in enumerate(messages):
        if "role" not in message or "content" not in message:
            return jsonify({"error": "Each message needs role and content"}), 400
        role = "model" if message["role"].lower() == "assistant" else "user"
        content = str(message["content"])
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
    except Exception as exc:
        return jsonify({"error": f"Failed to generate chat response: {exc}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=8000)
