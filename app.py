import os
import numpy as np
from PIL import Image
import tensorflow as tf

from flask import Flask, jsonify, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

MODEL_PATH = 'Model.hdf5'

# Load your trained model
print(" ** Model Loading **")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print(" ** Model Loaded **")

CLASS_LABELS = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']


def model_predict(img_path, model):
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x, verbose=0)
    scores = preds[0]
    class_index = int(np.argmax(scores))
    class_name = CLASS_LABELS[class_index].split('___')

    crop_scores = {}
    for index, score in enumerate(scores):
        crop = CLASS_LABELS[index].split('___')[0]
        crop_scores[crop] = crop_scores.get(crop, 0.0) + float(score)

    top3_crops = sorted(crop_scores.items(), key=lambda item: item[1], reverse=True)[:3]
    top3_crops_result = [
        {'crop': crop, 'confidence': round(score * 100, 2)}
        for crop, score in top3_crops
    ]

    return {
        'predicted_crop': class_name[0],
        'predicted_disease': class_name[1].title().replace('_', ' ') if len(class_name) > 1 else 'Unknown',
        'confidence': round(float(scores[class_index]) * 100, 2),
        'top3_crops': top3_crops_result
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method != 'POST':
        return jsonify({'error': 'Method not allowed'}), 405

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        basepath = os.path.dirname(__file__)
        upload_dir = os.path.join(basepath, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, secure_filename(f.filename))
        f.save(file_path)

        prediction = model_predict(file_path, model)
        return jsonify(prediction)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
