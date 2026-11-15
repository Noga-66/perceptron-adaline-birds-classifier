"""
app.py
======
Web (HTML) GUI for Lab 4 / Task 1.   Run:   python app.py   then open http://127.0.0.1:5000

This file is only the web layer (routes + JSON). All learning / evaluation code
lives in logic.py, all drawing code in plotting.py, and the page itself in
templates/index.html.
"""
import base64
import io
import os
import re

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from flask import Flask, jsonify, render_template, request

from logic import (ALGORITHMS, ALL_FEATURES, TrainedClassifier, load_birds, run_experiment)
from plotting import plot_classifier, plot_decision_boundary, plot_learning_curve

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
CLASS_PAIRS = ["A & B", "A & C", "B & C"]

app = Flask(__name__)
os.makedirs(MODELS_DIR, exist_ok=True)
DF = load_birds(os.path.join(BASE_DIR, "birds.csv"))

# Single-user state: the model currently shown in the page.
STATE = {"clf": None}


# ------------------------------------------------------------------ helpers
def fig_to_base64(fig: Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def describe(clf: TrainedClassifier) -> dict:
    return {"algorithm": clf.algorithm, "features": clf.features, "classes": clf.classes,
            "accuracy": clf.accuracy, "train_accuracy": clf.train_accuracy, "cm": clf.cm,
            "epochs_run": clf.epochs_run, "eta": clf.eta, "epochs": clf.epochs,
            "use_bias": clf.use_bias, "mse_threshold": clf.mse_threshold,
            "weights": clf.w, "bias": clf.b}


def saved_models() -> list:
    return sorted(f for f in os.listdir(MODELS_DIR) if f.endswith(".json"))


def error(msg, code=400):
    return jsonify({"error": msg}), code


# ------------------------------------------------------------------- routes
@app.route("/")
def index():
    return render_template("index.html", features=ALL_FEATURES, class_pairs=CLASS_PAIRS,
                           algorithms=list(ALGORITHMS), models=saved_models())


@app.post("/api/train")
def api_train():
    p = request.get_json(force=True)
    try:
        features = tuple(p["features"])
        classes = tuple(p["classes"].split(" & "))
        algo = p["algorithm"]
        eta, epochs = float(p["eta"]), int(p["epochs"])
        mse_th = float(p["mse_threshold"])
        use_bias = bool(p["use_bias"])
    except (KeyError, ValueError, TypeError):
        return error("Invalid input: check that eta / MSE threshold are numbers and epochs is an integer.")
    if len(features) != 2 or features[0] == features[1]:
        return error("Please choose two DIFFERENT features.")
    if algo not in ALGORITHMS or classes not in [tuple(c.split(" & ")) for c in CLASS_PAIRS]:
        return error("Invalid algorithm or classes.")
    if eta <= 0 or epochs <= 0:
        return error("Learning rate and epochs must be positive.")

    res = run_experiment(DF, features, classes, algo, eta=eta, epochs=epochs,
                         mse_threshold=mse_th, use_bias=use_bias)
    STATE["clf"] = TrainedClassifier.from_result(res)

    fig = Figure(figsize=(11, 4.2))
    ax1, ax2 = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)
    plot_decision_boundary(res, ax1, title=f"{algo}: {classes[0]} vs {classes[1]}")
    plot_learning_curve(res, ax2)
    fig.tight_layout()
    return jsonify({"plot": fig_to_base64(fig), **describe(STATE["clf"])})


@app.post("/api/classify")
def api_classify():
    clf = STATE["clf"]
    if clf is None:
        return error("Train or load a model first.")
    try:
        values = [float(v) for v in request.get_json(force=True)["values"]]
    except (KeyError, ValueError, TypeError):
        return error("Sample values must be numbers.")
    if len(values) != 2:
        return error("Two feature values are required.")
    return jsonify({"class": clf.classify(values)})


@app.post("/api/save")
def api_save():
    clf = STATE["clf"]
    if clf is None:
        return error("Nothing to save: train a model first.")
    name = clf.default_filename()
    clf.save(os.path.join(MODELS_DIR, name))
    return jsonify({"saved_as": name, "models": saved_models()})


@app.post("/api/load")
def api_load():
    name = str(request.get_json(force=True).get("name", ""))
    if not re.fullmatch(r"[\w\-]+\.json", name) or name not in saved_models():
        return error("Model file not found.", 404)
    clf = TrainedClassifier.load(os.path.join(MODELS_DIR, name))
    STATE["clf"] = clf

    fig = Figure(figsize=(6, 4.2))
    plot_classifier(clf, DF, fig.add_subplot(1, 1, 1),
                    title=f"{clf.algorithm} (loaded): {clf.classes[0]} vs {clf.classes[1]}")
    fig.tight_layout()
    return jsonify({"plot": fig_to_base64(fig), **describe(clf)})


if __name__ == "__main__":
    app.run(debug=False)
