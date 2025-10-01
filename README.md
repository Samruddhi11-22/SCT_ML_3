# SCT_ML_3: Cat vs Dog SVM Classifier

A full-stack web application using Support Vector Machine (SVM) to classify cat vs dog images from the Kaggle dataset.

## Features
- SVM with HOG features for image classification (69% accuracy on test set).
- Flask backend for image upload and prediction.
- Interactive frontend with confidence meter, animations, and game-like flow.

## Setup
1. Clone the repo: `git clone https://github.com/Samruddhi11-22/SCT_ML_3.git`
2. Create virtual environment: `python -m venv venv`
3. Activate: `.\venv\Scripts\Activate.ps1` (PowerShell)
4. Install dependencies: `pip install -r requirements.txt`
5. Run training (if needed): `python prepare_data.py`
6. Start the app: `python app.py`
7. Open: http://localhost:5000

## Demo
- Upload an image and follow the "Start Adventure" flow to get predictions with confidence scores.
- Example: Prediction: Dog (85%) 🐶

## Files
- `app.py`: Flask backend.
- `prepare_data.py`: SVM training script.
- `model.pkl`: Trained SVM model.
- `templates/index.html`: Interactive frontend.
- `requirements.txt`: Dependencies.
- `train_subset/`, `test_subset/`: Image datasets (300 cats, 300 dogs).

## License
MIT License
