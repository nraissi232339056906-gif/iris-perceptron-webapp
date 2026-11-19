
# Iris Perceptron Classifier — TP2 Web App

A Flask web application for the Iris classification TP. It includes:
- Interactive four-feature prediction form.
- NumPy Perceptron implemented from scratch.
- One-vs-Rest multiclass classification (three binary Perceptrons).
- Accuracy, macro Precision/Recall/F1 and confusion matrix.
- Training error chart and model weights/biases.
- Arabic RTL interface.

## Run locally
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000

## Deploy on Render
1. Upload this project to a GitHub repository.
2. In Render choose **New + → Web Service** and connect the repository.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`
5. Deploy. Render will provide a public URL.

## Important scientific note
The displayed multiclass metrics are recalculated by this app using a stratified 80/20 split (random_state=42), StandardScaler fitted only on the training set, and three NumPy Perceptrons trained for 50 epochs. They should not be assumed identical to a different notebook run unless its exact code, initialization, and training settings match.
The softmax percentages in the prediction panel are a display transformation of model scores, not calibrated probabilities.
