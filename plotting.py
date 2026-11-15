"""
plotting.py
===========
Matplotlib helpers (no GUI toolkit code). Functions draw on a given Axes so
they can be embedded in Tkinter or saved to PNG files.
"""
import numpy as np

from logic import ExperimentResult, decision_line_original_scale


def plot_decision_boundary(result: ExperimentResult, ax, title=None, show_test=True):
    """Scatter both classes and draw the line  W1*X1 + W2*X2 + b = 0."""
    ds, model = result.dataset, result.model
    colors = {-1: "tab:blue", 1: "tab:orange"}
    for lab, cls in zip((-1, 1), ds.classes):
        m = ds.y_train == lab
        ax.scatter(ds.X_train[m, 0], ds.X_train[m, 1], c=colors[lab], s=28, label=f"{cls} (train)")
        if show_test:
            m = ds.y_test == lab
            ax.scatter(ds.X_test[m, 0], ds.X_test[m, 1], c=colors[lab], s=45,
                       marker="x", label=f"{cls} (test)")

    all_x = np.vstack([ds.X_train, ds.X_test])
    pad = 0.05 * np.ptp(all_x, axis=0)
    x_lo, y_lo = all_x.min(axis=0) - pad
    x_hi, y_hi = all_x.max(axis=0) + pad

    a1, a2, c = decision_line_original_scale(model, ds.scaler)
    if abs(a2) > 1e-12:
        xs = np.array([x_lo, x_hi])
        ax.plot(xs, -(a1 * xs + c) / a2, "k-", lw=1.8, label="decision line")
    elif abs(a1) > 1e-12:
        ax.axvline(-c / a1, color="k", lw=1.8, label="decision line")

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(ds.features[0])
    ax.set_ylabel(ds.features[1])
    if title:
        ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="best")


def plot_learning_curve(result: ExperimentResult, ax):
    """MSE per epoch for Adaline, number of training errors per epoch for Perceptron."""
    if result.model.name == "Adaline":
        ax.plot(result.model.history["mse"])
        ax.set_ylabel("training MSE")
    else:
        ax.plot(result.model.history["errors"])
        ax.set_ylabel("# misclassified (train)")
    ax.set_xlabel("epoch")
    ax.set_title("Learning curve", fontsize=10)


def plot_classifier(clf, df, ax, title=None):
    """Plot a model loaded from disk: all samples of its two classes + its decision line."""
    colors = ["tab:blue", "tab:orange"]
    f1, f2 = clf.features
    pts = []
    for cls_name, col in zip(clf.classes, colors):
        sub = df[df["species"] == cls_name][[f1, f2]].to_numpy(float)
        pts.append(sub)
        ax.scatter(sub[:, 0], sub[:, 1], c=col, s=28, label=cls_name)
    all_x = np.vstack(pts)
    pad = 0.05 * np.ptp(all_x, axis=0)
    x_lo, y_lo = all_x.min(axis=0) - pad
    x_hi, y_hi = all_x.max(axis=0) + pad
    a1, a2, c = clf.decision_line()
    if abs(a2) > 1e-12:
        xs = np.array([x_lo, x_hi])
        ax.plot(xs, -(a1 * xs + c) / a2, "k-", lw=1.8, label="decision line")
    elif abs(a1) > 1e-12:
        ax.axvline(-c / a1, color="k", lw=1.8, label="decision line")
    ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(f1); ax.set_ylabel(f2)
    if title:
        ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="best")
