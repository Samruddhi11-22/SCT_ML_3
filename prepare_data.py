import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle
import random

def create_subset():
    folder = 'data'  # Use data/cats and data/dogs
    
    # Check if folders exist
    if not os.path.exists(os.path.join(folder, 'cats')) or not os.path.exists(os.path.join(folder, 'dogs')):
        raise FileNotFoundError(f"Folders {os.path.join(folder, 'cats')} or {os.path.join(folder, 'dogs')} not found.")
    
    cat_files = [f for f in os.listdir(os.path.join(folder, 'cats')) if f.endswith(('.jpg', '.png'))]
    dog_files = [f for f in os.listdir(os.path.join(folder, 'dogs')) if f.endswith(('.jpg', '.png'))]
    
    print(f"Found {len(cat_files)} cat images: {cat_files[:5]}")
    print(f"Found {len(dog_files)} dog images: {dog_files[:5]}")
    
    # Use 200 for training, 100 for testing
    random.seed(42)
    n_train = min(200, len(cat_files) - 1, len(dog_files) - 1)
    n_test = min(100, len(cat_files) - n_train, len(dog_files) - n_train)
    
    if n_train < 1 or n_test < 1:
        raise ValueError(f"Not enough images: Cats ({len(cat_files)}), Dogs ({len(dog_files)}). Need at least 2 per class.")
    
    os.makedirs('train_subset/cats', exist_ok=True)
    os.makedirs('train_subset/dogs', exist_ok=True)
    os.makedirs('test_subset/cats', exist_ok=True)
    os.makedirs('test_subset/dogs', exist_ok=True)
    
    train_cats = random.sample(cat_files, n_train)
    train_dogs = random.sample(dog_files, n_train)
    test_cats = random.sample([f for f in cat_files if f not in train_cats], n_test)
    test_dogs = random.sample([f for f in dog_files if f not in train_dogs], n_test)
    
    for f in train_cats:
        os.rename(os.path.join(folder, 'cats', f), os.path.join('train_subset', 'cats', f))
    for f in train_dogs:
        os.rename(os.path.join(folder, 'dogs', f), os.path.join('train_subset', 'dogs', f))
    for f in test_cats:
        os.rename(os.path.join(folder, 'cats', f), os.path.join('test_subset', 'cats', f))
    for f in test_dogs:
        os.rename(os.path.join(folder, 'dogs', f), os.path.join('test_subset', 'dogs', f))
    print(f"Subsets created: {n_train} training and {n_test} testing images per class.")

def extract_features(folder, use_hog=True):
    features = []
    labels = []
    for label in ['cats', 'dogs']:
        class_label = 0 if label == 'cats' else 1
        for img_file in os.listdir(os.path.join(folder, label)):
            img_path = os.path.join(folder, label, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Skipping invalid image: {img_path}")
                continue
            img = cv2.resize(img, (64, 64))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if use_hog:
                from skimage.feature import hog
                feat = hog(gray, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
            else:
                feat = gray.flatten()
            
            features.append(feat)
            labels.append(class_label)
    return np.array(features), np.array(labels)

def train_model(use_hog=True):
    X, y = extract_features('train_subset', use_hog)
    X_test, y_test = extract_features('test_subset', use_hog)
    
    if len(X) == 0 or len(X_test) == 0:
        raise ValueError("No valid images found in train_subset or test_subset.")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)
    
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm.fit(X_scaled, y)
    
    y_pred = svm.predict(X_test_scaled)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=['Cat', 'Dog']))
    
    with open('model.pkl', 'wb') as f:
        pickle.dump({'svm': svm, 'scaler': scaler, 'use_hog': use_hog}, f)
    print("Model saved as model.pkl")

if __name__ == "__main__":
    os.makedirs('data/cats', exist_ok=True)
    os.makedirs('data/dogs', exist_ok=True)
    create_subset()
    train_model(use_hog=True)