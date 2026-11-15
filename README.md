#  Single-Layer Perceptron & Adaline (Birds Dataset)

Neural Networks course, 2025–2026.

This project implements two single-neuron classifiers **from scratch (NumPy only)** — the **Perceptron** and **Adaline** — and applies them to the Birds dataset (3 species: A, B, C; 50 samples each). The user picks **two features** and **two classes**; the model is trained on 30 random samples per class and tested on the remaining 20 per class. A web GUI lets you train, visualise, save/load models and classify a single sample.

Live demo: https://nana66.pythonanywhere.com

<img width="1222" height="838" alt="Screenshot 2026-11-15 161957" src="https://github.com/user-attachments/assets/065d5de4-a8af-428a-9ef1-09de3327cfc9" />

<img width="1220" height="842" alt="Screenshot 2026-11-15 161943" src="https://github.com/user-attachments/assets/62d9b56b-9b59-49ea-9918-ccc3493d854e" />



## Workflow

1. **`Lab4_Task1_Training.ipynb`** runs all feature/class combinations for both algorithms, plots the decision lines, computes confusion matrices and accuracy, and **saves the trained models** to `models/` (JSON files).
2. **`app.py`** serves the HTML GUI, where you can train a new model or **load** any model saved by the notebook and classify a sample without retraining.

## Features

- Perceptron (signum activation, error-driven updates) and Adaline (linear activation, MSE cost, sample-by-sample gradient descent)
- Optional bias, configurable learning rate, number of epochs and MSE threshold (stopping criterion)
- Random, non-repeated 30 train / 20 test split per class
- Decision-line visualisation on the scatter plot of both classes + learning curve
- Confusion matrix and overall accuracy implemented manually (no scikit-learn)
- Save / load trained models (weights, bias, scaler statistics, parameters, results)
- Logic (`logic.py`) fully separated from the UI (`app.py`, `templates/index.html`)

## Dataset & preprocessing

| Item | Detail |
|---|---|
| Features | `gender`, `body_mass`, `beak_length`, `beak_depth`, `fin_length` |
| Classes | A, B, C (50 samples each) |
| Missing values | 6 missing `gender` values are filled with the most frequent gender of the same species — **no row is dropped** |
| `gender` encoding | female = 0, male = 1 |
| Scaling | Z-score standardisation, fitted on the training set only (needed because `body_mass` is in the thousands); the decision line is converted back to the original feature scale for plotting |

## Algorithms

For a sample `x` with desired output `d ∈ {-1, +1}`, net value `v = w·x + b` and error `e = d − y`:

- **Perceptron:** `y = signum(v)`; update `w ← w + η·e·x` (and `b ← b + η·e`). Stops when an epoch has no misclassified training sample, or after `m` epochs.
- **Adaline:** `y = v` (linear) during training; same update rule. After each epoch `MSE = (1/m) Σ ½ e²` is computed over all training samples; stops when `MSE < mse_threshold` or after `m` epochs.
- **Testing (both):** `y = signum(w·x + b)` → class ID.
- Decision line: `W1·X1 + W2·X2 + b = 0`.

## Results

Mean test accuracy over 10 random splits (η = 0.01, 200 epochs, with bias):

| Algorithm | Best combination | Accuracy | Worst combination | Accuracy |
|---|---|---|---|---|
| Perceptron | A vs B, `beak_length` + `fin_length` | 99.75 % | A vs C, `gender` + `beak_depth` | 50.25 % |
| Adaline | A vs B, `body_mass` + `beak_depth` | 100 % | A vs C, `gender` + `body_mass` | 49.00 % |

Pairs of morphological measurements (beak / fin / body mass) separate the classes well, while combinations that include `gender` for some class pairs are close to chance. Full per-combination tables and plots are in the notebook.

## Installation & usage

```bash
pip install -r requirements.txt

# 1) (optional) re-train and re-save the models
jupyter notebook            # open Lab4_Task1_Training.ipynb and Run All

# 2) start the GUI
python app.py               # open http://127.0.0.1:5000
```

### Using the GUI
1. Choose two different features, two classes, learning rate, epochs, MSE threshold, bias and algorithm, then press **Train & Test**.
2. The page shows the scatter plot with the decision line, the learning curve, the confusion matrix and the overall accuracy.
3. **Save current model** stores it in `models/`; **Load selected model** restores any saved model.
4. Enter feature values under **Classify a single sample** to get the predicted class.

## Project structure

```
├── Lab4_Task1_Training.ipynb   # training, evaluation, plots, saves models
├── app.py                      # Flask web layer (routes only)
├── templates/index.html        # HTML GUI
├── logic.py                    # loading, preprocessing, split, Perceptron, Adaline,
│                               # confusion matrix, save/load model
├── plotting.py                 # decision-boundary and learning-curve plots
├── models/                     # saved trained models (JSON)
├── birds.csv                   # dataset
└── requirements.txt
```
