"""
Model Evaluation Script
Calculates Accuracy, Precision, and Recall metrics for the trained model.

Usage:
    python evaluate.py --dataset_path <path_to_dataset>
    
Example:
    python evaluate.py --dataset_path "./new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/valid"
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
from PIL import Image
import argparse
import json


# Class labels (same as in app.py)
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


def load_model():
    """Load the trained model"""
    print("Loading model...")
    model = tf.keras.models.load_model('Model.hdf5', compile=False)
    return model


def preprocess_image(img_path):
    """Load and preprocess a single image"""
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    x = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(x, axis=0)


def get_predictions_from_directory(model, dataset_path):
    """
    Get predictions for all images in a directory structure.
    Directory structure should be: dataset_path/class_name/*.jpg
    """
    true_labels = []
    pred_labels = []
    
    print(f"Processing images from {dataset_path}...")
    
    for class_idx, class_name in enumerate(CLASS_LABELS):
        class_dir = os.path.join(dataset_path, class_name)
        
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} not found")
            continue
        
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(image_files) == 0:
            print(f"Warning: No images found in {class_dir}")
            continue
        
        print(f"Processing {len(image_files)} images from {class_name}...")
        
        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            try:
                x = preprocess_image(img_path)
                preds = model.predict(x, verbose=0)
                pred_class_idx = int(np.argmax(preds[0]))
                
                true_labels.append(class_idx)
                pred_labels.append(pred_class_idx)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    return np.array(true_labels), np.array(pred_labels)


def calculate_metrics(true_labels, pred_labels):
    """Calculate Accuracy, Precision, and Recall"""
    
    # Accuracy: Overall correctness
    accuracy = accuracy_score(true_labels, pred_labels)
    
    # Precision: Of all positive predictions, how many were correct?
    precision_weighted = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
    precision_macro = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
    
    # Recall: Of all actual positives, how many did we find?
    recall_weighted = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
    recall_macro = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision_weighted': precision_weighted,
        'precision_macro': precision_macro,
        'recall_weighted': recall_weighted,
        'recall_macro': recall_macro
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate crop disease detection model')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to dataset directory (contains class subdirectories)')
    parser.add_argument('--output_json', type=str, default='evaluation_results.json',
                        help='Output file for metrics (JSON format)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path {args.dataset_path} does not exist")
        return
    
    # Load model
    model = load_model()
    
    # Get predictions
    true_labels, pred_labels = get_predictions_from_directory(model, args.dataset_path)
    
    if len(true_labels) == 0:
        print("Error: No images were processed")
        return
    
    # Calculate metrics
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    
    metrics = calculate_metrics(true_labels, pred_labels)
    
    print(f"\n✓ Accuracy: {metrics['accuracy']:.4f}")
    print(f"  → Overall correctness of all predictions\n")
    
    print(f"✓ Precision (Weighted): {metrics['precision_weighted']:.4f}")
    print(f"  → Of all positive predictions, how many were correct?")
    print(f"✓ Precision (Macro): {metrics['precision_macro']:.4f}\n")
    
    print(f"✓ Recall (Weighted): {metrics['recall_weighted']:.4f}")
    print(f"  → Of all actual positives, how many did we find?")
    print(f"✓ Recall (Macro): {metrics['recall_macro']:.4f}\n")
    
    print("="*60)
    print(f"Total images evaluated: {len(true_labels)}")
    print("="*60)
    
    # Detailed report
    print("\nDetailed Classification Report:\n")
    print(classification_report(true_labels, pred_labels, target_names=CLASS_LABELS, zero_division=0))
    
    # Save to JSON
    metrics['total_images'] = int(len(true_labels))
    metrics['correct_predictions'] = int(np.sum(true_labels == pred_labels))
    metrics['incorrect_predictions'] = int(len(true_labels) - metrics['correct_predictions'])
    
    with open(args.output_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {args.output_json}")


if __name__ == '__main__':
    main()
