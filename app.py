
from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import Perceptron

app = Flask(__name__)
app.jinja_env.globals.update(enumerate=enumerate)

class PerceptronScratch:
    """Binary Perceptron implemented with NumPy."""
    def __init__(self, eta=0.1, epochs=50, random_state=42):
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        self.w = rng.normal(0, 0.01, X.shape[1])
        self.b = 0.0
        self.errors_ = []
        for _ in range(self.epochs):
            errors = 0
            # Shuffle samples at every epoch, as in the notebook implementation.
            order = rng.permutation(len(y))
            for idx in order:
                xi, yi = X[idx], y[idx]
                pred = 1 if np.dot(xi, self.w) + self.b >= 0 else 0
                error = yi - pred
                if error != 0:
                    self.w += self.eta * error * xi
                    self.b += self.eta * error
                    errors += 1
            self.errors_.append(errors)
        return self

    def score_raw(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        return (self.score_raw(X) >= 0).astype(int)


class OvRPerceptron:
    def __init__(self, eta=0.1, epochs=50):
        self.eta, self.epochs = eta, epochs

    def fit(self, X, y):
        self.models_ = []
        for k in range(3):
            binary_y = (y == k).astype(int)
            model = PerceptronScratch(self.eta, self.epochs, 42 + k)
            model.fit(X, binary_y)
            self.models_.append(model)
        return self

    def scores(self, X):
        return np.column_stack([m.score_raw(X) for m in self.models_])

    def predict(self, X):
        return np.argmax(self.scores(X), axis=1)


iris = load_iris()
X, y = iris.data, iris.target
names = list(iris.target_names)
feature_names = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = OvRPerceptron(eta=0.1, epochs=50).fit(X_train_s, y_train)
test_pred = model.predict(X_test_s)
cm = confusion_matrix(y_test, test_pred)
metrics = {
    "accuracy": round(float(accuracy_score(y_test, test_pred)), 4),
    "precision": round(float(precision_score(y_test, test_pred, average="macro", zero_division=0)), 4),
    "recall": round(float(recall_score(y_test, test_pred, average="macro", zero_division=0)), 4),
    "f1": round(float(f1_score(y_test, test_pred, average="macro", zero_division=0)), 4),
}
# Train binary scratch model for the TP's Setosa vs Versicolor experiment.
mask = y < 2
Xb, yb = X[mask][:, [2, 3]], y[mask]
Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    Xb, yb, test_size=0.20, random_state=42, stratify=yb
)
binary_model = PerceptronScratch(eta=0.1, epochs=20).fit(Xb_train, yb_train)
binary_accuracy = float(accuracy_score(yb_test, binary_model.predict(Xb_test)))

@app.route("/")
def index():
    return render_template(
        "index.html",
        names=names,
        flower_images=[
            "https://www.re-natur.de/media/catalog/product/cache/e42b3e3e6979353f5fe1d4a80fd6f01b/4/0/40271_iris_setosa.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg"
        ],
        feature_names=feature_names,
        metrics=metrics,
        cm=cm.tolist(),
        accuracy_percent=round(metrics["accuracy"] * 100, 1),
        binary_accuracy_percent=round(binary_accuracy * 100, 1),
        epochs=list(range(1, 51)),
        errors=[sum(m.errors_[i] for m in model.models_) for i in range(50)]
    )

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    vals = np.array([[float(data[k]) for k in ["sepal_length", "sepal_width", "petal_length", "petal_width"]]])
    vals_s = scaler.transform(vals)
    scores = model.scores(vals_s)[0]
    pred = int(np.argmax(scores))
    # Scores are not probabilities; softmax is only a normalized display of relative model scores.
    shifted = scores - np.max(scores)
    exp_scores = np.exp(shifted)
    display = exp_scores / exp_scores.sum()
    return jsonify({
        "class_id": pred,
        "class_name": names[pred].capitalize(),
        "image": [
            "https://www.re-natur.de/media/catalog/product/cache/e42b3e3e6979353f5fe1d4a80fd6f01b/4/0/40271_iris_setosa.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg"
        ][pred],
        "scores": [round(float(v), 4) for v in scores],
        "display_scores": [round(float(v * 100), 1) for v in display],
        "weights": [[round(float(v), 4) for v in m.w] for m in model.models_],
        "biases": [round(float(m.b), 4) for m in model.models_]
    })

@app.route("/api/model")
def model_info():
    return jsonify({
        "classes": names,
        "features": feature_names,
        "weights": [[round(float(v), 4) for v in m.w] for m in model.models_],
        "biases": [round(float(m.b), 4) for m in model.models_],
        "metrics": metrics,
        "confusion_matrix": cm.tolist()
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
