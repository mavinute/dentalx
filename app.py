from ultralytics import YOLO
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from waitress import serve
from PIL import Image
import os

# --------------------------------------
# CONFIGURAÇÕES
# --------------------------------------

app = Flask(__name__)

# Habilita CORS apenas para o endpoint /detect
CORS(
    app,
    resources={r"/detect": {"origins": "*"}}
)

MODEL_PATH = "best.pt"

# Carrega o modelo uma única vez
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"O arquivo '{MODEL_PATH}' não foi encontrado. "
        f"Coloque um modelo YOLO válido na raiz do projeto."
    )

model = YOLO(MODEL_PATH)

# --------------------------------------
# ROTAS
# --------------------------------------

@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "service": "Dentiscan AI",
        "status": "running"
    })

@app.route("/detect", methods=["POST"])
def detect():
    """
    Recebe imagem via multipart/form-data
    Retorna bounding boxes no formato:
    [[x1, y1, x2, y2, label, confidence], ...]
    """

    if "image_file" not in request.files:
        return jsonify({"error": "Nenhuma imagem enviada"}), 400

    file = request.files["image_file"]

    try:
        boxes = detect_objects_on_image(file.stream)
        return jsonify(boxes)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------------------------
# FUNÇÃO PRINCIPAL DE DETECÇÃO
# --------------------------------------

def detect_objects_on_image(buf):
    img = Image.open(buf).convert("RGB")

    results = model.predict(img)
    result = results[0]

    output = []

    for box in result.boxes:
        x1, y1, x2, y2 = [int(x) for x in box.xyxy[0].tolist()]
        class_id = int(box.cls[0])
        prob = float(box.conf[0])

        output.append([
            x1,
            y1,
            x2,
            y2,
            result.names[class_id],
            f"{prob * 100:.2f}%"
        ])

    return output

# --------------------------------------
# INICIALIZAÇÃO DO SERVIDOR
# --------------------------------------

if __name__ == "__main__":
    print("Dentiscan AI rodando na porta 8080")
    serve(app, host="0.0.0.0", port=8080)
