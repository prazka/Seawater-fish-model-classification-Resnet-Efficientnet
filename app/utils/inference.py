from PIL import Image
import numpy as np
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess_input
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
from tensorflow.keras.models import load_model

model_resnet50 = 'app\model\model_eff_aug1.4.h5'  # Ganti dengan path model ResNet50 Anda
model_efficientnet = 'app\model\model_resnet_aug1.4.h5'  # Ganti dengan path model EfficientNet Anda

labels = {'Ikan_Badut': 0,
    'Ikan_Barakuda': 1,
    'Ikan_ekor_kuning': 2,
    'Ikan_kakapmerah': 3,
    'Ikan_kerapu': 4,
    'Ikan_tenggiri': 5,
    'Ikan_tongkol': 6}  


def preprocess_image_for_resnet50(pil_image: Image.Image) -> np.ndarray:
    """
    Melakukan preprocessing gambar agar sesuai dengan input ResNet50.
    """
    pil_image = pil_image.resize((224, 224))
    pil_image = pil_image.convert("RGB")
    img_array = np.array(pil_image) 
    img_array_expanded = np.expand_dims(img_array, axis=0)
    processed_img = resnet_preprocess_input(img_array_expanded)
    return processed_img

def preprocess_image_for_efficientnet(pil_image: Image.Image) -> np.ndarray:
    """
    Melakukan preprocessing gambar agar sesuai dengan input EfficientNet.
    """
    pil_image = pil_image.resize((224, 224))
    pil_image = pil_image.convert("RGB")
    img_array = np.array(pil_image)  # Hasilnya array (224, 224, 3)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    processed_img = efficientnet_preprocess_input(img_array_expanded)
    return processed_img

def predict_image(pil_image: Image.Image, model_type: str) -> str:
    """
    Melakukan prediksi kelas gambar menggunakan model ResNet50.
    """
    if model_type == "resnet":
        model = load_model(model_resnet50)
        input_tensor = preprocess_image_for_resnet50(pil_image)
    elif model_type == "efficientnet":
        model = load_model(model_efficientnet)
        input_tensor = preprocess_image_for_efficientnet(pil_image)
    else:
        raise ValueError("model_type must be 'resnet' or 'efficientnet'")
    predictions = model.predict(input_tensor)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = float(np.max(predictions, axis=1)[0])
    return labels[predicted_class_index], confidence_score

