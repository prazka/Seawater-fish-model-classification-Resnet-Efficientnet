from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from app.utils.inference import predict_image
import os
from PIL import Image

predict_route = Blueprint("predict_route", __name__, url_prefix="/api")

@predict_route.route("/home", methods=["GET"])
def index():
    return jsonify({"message": "Welcome to the Fish Image Classification API"}), 200

@predict_route.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(image_file.filename)
    img = Image.open(image_file.stream).convert("RGB")
    
    model_type = request.form.get("model_type")
    if model_type not in ["resnet", "efficientnet"]:
        return jsonify({"error": "model_type must be 'resnet' or 'efficientnet'"}), 400

    try:
        img = Image.open(image_file.stream).convert("RGB")
        prediction, confidence = predict_image(img, model_type)
        return jsonify({"prediction": prediction, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": str(e)}), 500