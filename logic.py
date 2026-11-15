"""
logic.py
========
All non-UI logic for Lab 4 / Task 1 (Birds dataset):
data loading, preprocessing, train/test split, Perceptron, Adaline,
confusion matrix, accuracy and decision-line computation.

Nothing in this module imports a GUI library, so it can be reused from the
GUI (gui.py), from scripts (experiments.py) or from tests.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ALL_FEATURES = ["gender", "body_mass", "beak_length", "beak_depth", "fin_length"]
ALL_CLASSES = ["A", "B", "C"]


# --------------------------------------------------------------------------- #
# Data loading & preprocessing
# --------------------------------------------------------------------------- #
def load_birds(path: str = "birds.csv") -> pd.DataFrame:
    """Load the dataset WITHOUT dropping any row.

    * Missing ``gender`` values are filled with the most frequent gender of the
      same species.
    * ``gender`` is encoded numerically: female = 0, male = 1.
    * The ``bird category`` column is renamed to ``species``.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"bird category": "species"})
    df["species"] = df["species"].str.strip()

    gender = df["gender"].astype("string").str.strip().str.lower()
    mode_per_species = gender.groupby(df["species"]).transform(lambda s: s.mode().iloc[0])
    gender = gender.fillna(mode_per_species)
    df["gender"] = gender.map({"female": 0.0, "male": 1.0}).astype(float)
    return df


@dataclass
class Standardizer:
    """Z-score scaler, fitted on the training set only."""
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


@dataclass
class Dataset:
    """One binary-classification problem. Labels: classes[0] -> -1, classes[1] -> +1."""
    features: tuple
    classes: tuple
    X_train: np.ndarray          # original (un-scaled) values
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: Standardizer


def make_dataset(df: pd.DataFrame, features, classes, n_train: int = 30,
                 rng: np.random.Generator | None = None) -> Dataset:
    """Randomly pick ``n_train`` non-repeated samples per class for training;
    the remaining samples (20 per class) are used for testing."""
    rng = np.random.default_rng() if rng is None else rng
    Xtr, ytr, Xte, yte = [], [], [], []
    for label, cls in zip((-1, 1), classes):
        sub = df[df["species"] == cls][list(features)].to_numpy(float)
        idx = rng.permutation(len(sub))
        tr, te = idx[:n_train], idx[n_train:]
        Xtr.append(sub[tr]); ytr += [label] * len(tr)
        Xte.append(sub[te]); yte += [label] * len(te)
    X_train = np.vstack(Xtr)
    return Dataset(tuple(features), tuple(classes), X_train, np.array(ytr),
                   np.vstack(Xte), np.array(yte), Standardizer().fit(X_train))


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def signum(v):
    """Hard limiter: returns +1 for v >= 0, otherwise -1."""
    return np.where(v >= 0, 1, -1)


class LinearNeuron:
    """Single neuron (n inputs + optional bias). Subclasses define the training
    activation and the stopping rule."""
    name = "LinearNeuron"

    def __init__(self, eta=0.01, epochs=100, use_bias=True, mse_threshold=0.0, seed=None):
        self.eta = eta
        self.epochs = epochs
        self.use_bias = use_bias
        self.mse_threshold = mse_threshold
        self.rng = np.random.default_rng(seed)
        self.w = None
        self.b = 0.0
        self.history = {"mse": [], "errors": []}

    def _activation(self, v):
        raise NotImplementedError

    def _should_stop(self, mse, n_errors) -> bool:
        raise NotImplementedError

    def net_input(self, X):
        return X @ self.w + self.b

    def predict(self, X):
        """Class ID in {-1, +1}; the signum activation is always used for testing."""
        return signum(self.net_input(X))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = self.rng.uniform(-0.1, 0.1, n_features)               # small random weights
        self.b = self.rng.uniform(-0.1, 0.1) if self.use_bias else 0.0  # small random bias
        self.history = {"mse": [], "errors": []}

        for _ in range(self.epochs):
            for i in self.rng.permutation(n_samples):
                v = self.net_input(X[i])                # net value
                out = self._activation(v)               # actual output
                error = y[i] - out                      # d - y
                self.w = self.w + self.eta * error * X[i]
                if self.use_bias:
                    self.b = self.b + self.eta * error  # bias input is a constant 1

            # per-epoch diagnostics with the updated weights
            v_all = self.net_input(X)
            e_all = y - self._activation(v_all)
            mse = float(np.mean(0.5 * e_all ** 2))
            n_err = int(np.sum(signum(v_all) != y))
            self.history["mse"].append(mse)
            self.history["errors"].append(n_err)
            if self._should_stop(mse, n_err):
                break
        return self


class Perceptron(LinearNeuron):
    """Signum activation; stops when an epoch has no misclassified sample."""
    name = "Perceptron"

    def _activation(self, v):
        return signum(v)

    def _should_stop(self, mse, n_errors):
        return n_errors == 0


class Adaline(LinearNeuron):
    """Linear activation + MSE cost; stops when MSE < mse_threshold."""
    name = "Adaline"

    def _activation(self, v):
        return v

    def _should_stop(self, mse, n_errors):
        return mse < self.mse_threshold


ALGORITHMS = {"Perceptron": Perceptron, "Adaline": Adaline}


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def confusion_matrix(y_true, y_pred, labels=(-1, 1)) -> np.ndarray:
    """Manual confusion matrix. Rows = actual, columns = predicted."""
    idx = {lab: k for k, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    return cm


def accuracy_from_cm(cm: np.ndarray) -> float:
    return float(np.trace(cm) / cm.sum())


def decision_line_original_scale(model: LinearNeuron, scaler: Standardizer):
    """Weights were learned on standardized inputs; convert them back so the
    line a1*x1 + a2*x2 + c = 0 can be drawn on the raw feature axes."""
    w1, w2 = model.w
    m1, m2 = scaler.mean
    s1, s2 = scaler.std
    a1, a2 = w1 / s1, w2 / s2
    c = model.b - w1 * m1 / s1 - w2 * m2 / s2
    return a1, a2, c


# --------------------------------------------------------------------------- #
# One full experiment
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentResult:
    dataset: Dataset
    model: LinearNeuron
    cm: np.ndarray
    accuracy: float
    train_accuracy: float


def run_experiment(df, features, classes, algorithm, eta, epochs, mse_threshold=0.0,
                   use_bias=True, seed=None) -> ExperimentResult:
    """Split -> scale -> train -> test. Returns everything needed for display."""
    if len(set(features)) != 2:
        raise ValueError("Please choose two different features.")
    if len(set(classes)) != 2:
        raise ValueError("Please choose two different classes.")
    rng = np.random.default_rng(seed)
    ds = make_dataset(df, features, classes, n_train=30, rng=rng)
    model = ALGORITHMS[algorithm](eta=eta, epochs=epochs, use_bias=use_bias,
                                  mse_threshold=mse_threshold,
                                  seed=None if seed is None else seed + 1)
    X_train = ds.scaler.transform(ds.X_train)
    model.fit(X_train, ds.y_train)

    y_pred = model.predict(ds.scaler.transform(ds.X_test))
    cm = confusion_matrix(ds.y_test, y_pred)
    train_acc = float(np.mean(model.predict(X_train) == ds.y_train))
    return ExperimentResult(ds, model, cm, accuracy_from_cm(cm), train_acc)


def classify_sample(result: ExperimentResult, sample) -> str:
    """Classify ONE raw sample (original units, order = selected features)."""
    x = result.dataset.scaler.transform(np.asarray(sample, float).reshape(1, -1))
    label = int(result.model.predict(x)[0])
    return result.dataset.classes[0] if label == -1 else result.dataset.classes[1]


# --------------------------------------------------------------------------- #
# Saving / loading a trained model
# --------------------------------------------------------------------------- #
@dataclass
class TrainedClassifier:
    """Everything needed to classify new samples WITHOUT retraining.

    It stores the learned weights/bias, the scaler statistics, the selected
    features/classes, the hyper-parameters and the test results. It can be
    saved to / loaded from a plain JSON file.
    """
    algorithm: str
    features: list
    classes: list
    w: list
    b: float
    mean: list
    std: list
    eta: float
    epochs: int
    epochs_run: int
    use_bias: bool
    mse_threshold: float
    cm: list
    accuracy: float
    train_accuracy: float

    # ---- construction ----------------------------------------------------
    @classmethod
    def from_result(cls, res: "ExperimentResult") -> "TrainedClassifier":
        m, ds = res.model, res.dataset
        return cls(algorithm=m.name, features=list(ds.features), classes=list(ds.classes),
                   w=[float(v) for v in m.w], b=float(m.b),
                   mean=[float(v) for v in ds.scaler.mean], std=[float(v) for v in ds.scaler.std],
                   eta=float(m.eta), epochs=int(m.epochs), epochs_run=len(m.history["mse"]),
                   use_bias=bool(m.use_bias), mse_threshold=float(m.mse_threshold),
                   cm=res.cm.tolist(), accuracy=float(res.accuracy),
                   train_accuracy=float(res.train_accuracy))

    # ---- inference -------------------------------------------------------
    def classify(self, sample) -> str:
        """Classify ONE raw sample (original units, order = self.features)."""
        x = (np.asarray(sample, float) - np.array(self.mean)) / np.array(self.std)
        v = float(x @ np.array(self.w) + self.b)
        return self.classes[1] if v >= 0 else self.classes[0]

    def decision_line(self):
        """(a1, a2, c) for the line a1*x1 + a2*x2 + c = 0 on the raw feature axes."""
        w1, w2 = self.w
        (m1, m2), (s1, s2) = self.mean, self.std
        return w1 / s1, w2 / s2, self.b - w1 * m1 / s1 - w2 * m2 / s2

    # ---- persistence -----------------------------------------------------
    def default_filename(self) -> str:
        """e.g. Adaline_beak_length_fin_length_AC.json"""
        return f"{self.algorithm}_{self.features[0]}_{self.features[1]}_{self.classes[0]}{self.classes[1]}.json"

    def save(self, path: str) -> None:
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainedClassifier":
        import json
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))
