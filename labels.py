"""
Class labels shared by the Flask app, severity module, treatment engine and
benchmarking framework.

The order MUST match the output layer of Model.hdf5. It is the alphabetical
(sorted) order of the PlantVillage class directory names, which is the order
Keras' ``flow_from_directory`` assigned during training in
Crop-Diseases-Detection.ipynb. The same list is duplicated in app.py's original
version and evaluate.py; tests/test_labels.py checks that they stay identical.
"""

CLASS_LABELS = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

NUM_CLASSES = len(CLASS_LABELS)
LABEL_SEPARATOR = '___'


def split_label(class_label):
    """Return (crop, disease) parts of a PlantVillage label, e.g. ('Tomato', 'Late_blight')."""
    parts = class_label.split(LABEL_SEPARATOR)
    crop = parts[0]
    disease = parts[1] if len(parts) > 1 else 'Unknown'
    return crop, disease


def is_healthy(class_label):
    return split_label(class_label)[1].lower() == 'healthy'


def display_crop_name(class_label):
    """'Pepper,_bell___Bacterial_spot' -> 'Pepper, bell'"""
    return split_label(class_label)[0].replace('_', ' ').strip()


def display_disease_name(class_label):
    """Human-readable disease name using the same transform as the original app
    (title-case, underscores to spaces), with stray whitespace removed."""
    disease = split_label(class_label)[1]
    return ' '.join(disease.title().replace('_', ' ').split())


def class_index(class_label):
    return CLASS_LABELS.index(class_label)
