"""Task 3: classic ML threat classification (Logistic Regression and Decision Tree)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from common import OUT, RANDOM_STATE, evaluate, load_data, make_preprocessor, save_result, split

df = load_data()
X_train, X_test, y_train, y_test = split(df)
print(f"train={len(X_train)} test={len(X_test)} phishing share train={y_train.mean():.3f} test={y_test.mean():.3f}")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    # Depth limit keeps the tree readable and stops it memorising the training set.
    "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=RANDOM_STATE),
}

results = {}
fitted = {}
for name, clf in models.items():
    pipe = Pipeline([("prep", make_preprocessor(X_train)), ("clf", clf)])
    results[name] = evaluate(name, pipe, X_train, X_test, y_train, y_test)
    fitted[name] = pipe

# Explainability bonus: which encoded features push Logistic Regression towards "phishing"?
lr = fitted["Logistic Regression"]
names = lr.named_steps["prep"].get_feature_names_out()
coef = pd.Series(lr.named_steps["clf"].coef_[0], index=[n.split("__", 1)[1] for n in names])
top = coef.reindex(coef.abs().sort_values(ascending=False).index).head(12)
fig, ax = plt.subplots(figsize=(7, 5))
top.sort_values().plot.barh(ax=ax, color=["#DD5555" if v > 0 else "#4C72B0" for v in top.sort_values()])
ax.set(title="Logistic Regression: strongest weights\n(red = pushes towards phishing)", xlabel="coefficient")
fig.tight_layout()
fig.savefig(OUT / "lr_coefficients.png", dpi=150)

save_result("ml", results)
save_result("lr_top_weights", {k: round(float(v), 3) for k, v in top.items()})
