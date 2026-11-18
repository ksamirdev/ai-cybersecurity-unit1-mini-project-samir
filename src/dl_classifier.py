"""Task 4: simple neural network (MLPClassifier) and ML vs DL comparison.

Run ml_classifier.py first - this script reads its results to build the comparison.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from common import OUT, RANDOM_STATE, RESULTS_FILE, evaluate, load_data, make_preprocessor, save_result, split

df = load_data()
X_train, X_test, y_train, y_test = split(df)


def mlp():
    # Two hidden layers, ReLU + Adam. Early stopping holds out 10% of the training
    # data and stops when validation score stops improving (limits overfitting).
    return MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", alpha=1e-4,
                         max_iter=500, early_stopping=True, n_iter_no_change=15, random_state=RANDOM_STATE)


pipe = Pipeline([("prep", make_preprocessor(X_train)), ("clf", mlp())])
res = evaluate("Neural Network", pipe, X_train, X_test, y_train, y_test)
res["epochs"] = int(pipe.named_steps["clf"].n_iter_)
res["parameters"] = int(sum(w.size for w in pipe.named_steps["clf"].coefs_) +
                        sum(b.size for b in pipe.named_steps["clf"].intercepts_))
print(f"MLP stopped after {res['epochs']} epochs, {res['parameters']} trainable parameters")

# Training curve
clf = pipe.named_steps["clf"]
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(clf.loss_curve_, label="training loss")
ax2 = ax.twinx()
ax2.plot(clf.validation_scores_, color="#DD5555", label="validation accuracy")
ax.set(title="MLP training curve", xlabel="epoch", ylabel="loss")
ax2.set_ylabel("validation accuracy")
fig.legend(loc="center right", bbox_to_anchor=(0.88, 0.5))
fig.tight_layout()
fig.savefig(OUT / "mlp_training_curve.png", dpi=150)

# ---- Comparison -----------------------------------------------------------
all_results = json.loads(RESULTS_FILE.read_text())["ml"]
all_results["Neural Network"] = res

# 5-fold cross-validation on the full dataset: one test split can be lucky,
# the mean +/- std across folds is a fairer comparison.
X, y = df.drop(columns="phishing"), df["phishing"]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=RANDOM_STATE),
    "Neural Network": mlp(),
}
for name, m in cv_models.items():
    scores = cross_val_score(Pipeline([("prep", make_preprocessor(X)), ("clf", m)]), X, y, cv=cv, scoring="f1")
    all_results[name]["cv_f1_mean"] = float(scores.mean())
    all_results[name]["cv_f1_std"] = float(scores.std())
    print(f"{name:<20} 5-fold F1 = {scores.mean():.4f} +/- {scores.std():.4f}")

table = pd.DataFrame(all_results).T[["accuracy", "precision", "recall", "f1", "cv_f1_mean", "train_seconds", "predict_ms"]]
table.to_csv(OUT / "model_comparison.csv")
print(table.round(4))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [2, 1]})
metrics = ["accuracy", "precision", "recall", "f1"]
x = np.arange(len(metrics))
w = 0.26
colors = ["#4C72B0", "#55A868", "#C44E52"]
for i, (name, row) in enumerate(table.iterrows()):
    bars = axes[0].bar(x + (i - 1) * w, row[metrics], w, label=name, color=colors[i])
    axes[0].bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
axes[0].set(xticks=x, xticklabels=[m.capitalize() for m in metrics], ylim=(0.88, 0.98), title="Detection metrics (test set)")
axes[0].legend(loc="upper center", ncols=3, fontsize=8, bbox_to_anchor=(0.5, -0.08), frameon=False)
bars = axes[1].bar(table.index, table["train_seconds"], color=colors)
axes[1].bar_label(bars, fmt="%.3f s", fontsize=8, padding=2)
axes[1].set(title="Training time", ylabel="seconds")
axes[1].margins(y=0.15)
axes[1].tick_params(axis="x", rotation=15)
fig.tight_layout()
fig.savefig(OUT / "model_comparison.png", dpi=150)

save_result("dl", res)
save_result("comparison", all_results)
