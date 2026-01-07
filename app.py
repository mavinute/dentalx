from ultralytics import YOLO
from flask import Flask, request, jsonify
from flask_cors import CORS
from waitress import serve
from PIL import Image
import os

# --------------------------------------
# APP
# --------------------------------------

app = Flask(__name__)

# 🔥 CORS GLOBAL (obrigatório no Render)
CORS(
    app,
    origins="*",
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "OPTIONS"]
)

# --------------------------------------
# MODELO
# --------------------------------------

MODEL_PATH = "best.pt"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modelo '{MODEL_PATH}' não encontrado.")

model = YOLO(MODEL_PATH)

# --------------------------------------
# ROTAS
# --------------------------------------

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "service": "Dentiscan AI",
        "status": "online"
    })

@app.route("/detect", methods=["POST", "OPTIONS"])
def detect():
    # 🔥 Preflight (CORS)
    if request.method == "POST":
        return "", 200

    if "image_file" not in request.files:
        return jsonify({"error": "Nenhuma imagem enviada"}), 400

    file = request.files["image_file"]

    try:
        results = detect_objects_on_image(file.stream)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------------------------
# DETECÇÃO
# --------------------------------------

def detect_objects_on_image(buf):
    img = Image.open(buf).convert("RGB")

    results = model.predict(img)
    result = results[0]

    output = []

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        class_id = int(box.cls[0])
        conf = float(box.conf[0])

        output.append([
            x1,
            y1,
            x2,
            y2,
            result.names[class_id],
            f"{conf * 100:.2f}%"
        ])

    return output

# --------------------------------------
# SERVER
# --------------------------------------

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8080)
