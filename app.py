import os
from flask import request, Flask, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from waitress import serve
from PIL import Image

app = Flask(__name__)

CORS(
    app,
    resources={r"/detect": {"origins": "*"}},
)

MODEL_PATH = "./best.pt"
model = YOLO(MODEL_PATH)

@app.route("/")
def root():
    return jsonify({"status": "ok"})

@app.route("/detect", methods=["POST"])
def detect():
    if "image_file" not in request.files:
        return jsonify({"error": "Nenhuma imagem enviada"}), 400

    try:
        img = Image.open(request.files["image_file"].stream).convert("RGB")
        results = model.predict(img)
        result = results[0]

        output = []
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            output.append([
                x1, y1, x2, y2,
                result.names[cls],
                f"{conf * 100:.2f}%"
            ])

        return jsonify(output)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    serve(app, host="0.0.0.0", port=port)
