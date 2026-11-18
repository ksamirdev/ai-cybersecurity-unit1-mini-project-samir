"""Task 5: ethical risk simulation.

Part A - False positives block legitimate users:
    sweep the decision threshold and count how many legitimate websites
    (i.e. real businesses and their users) get blocked vs phishing sites missed.

Part B - Biased data leads to unfair decisions:
    young domains (registered < 6 months, `age_of_domain == -1`) are often new
    small businesses. If the training data barely contains *legitimate* young
    domains, the model learns "young domain = phishing" and unfairly blocks them.
    We compare a model trained on fair data vs one trained on biased data, then
    try two mitigations.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from common import OUT, RANDOM_STATE, load_data, make_preprocessor, save_result, split

df = load_data()
X_train, X_test, y_train, y_test = split(df)


def model():
    return Pipeline([("prep", make_preprocessor(X_train)),
                     ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))])


def rates(y_true, proba, t):
    pred = (proba >= t).astype(int)
    legit, phish = y_true == 0, y_true == 1
    return {
        "threshold": t,
        "fpr": float(pred[legit].mean()),          # legitimate sites wrongly blocked
        "fnr": float(1 - pred[phish].mean()),      # phishing sites that slip through
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
    }


# ---- Part A: threshold vs false positives ---------------------------------
base = model().fit(X_train, y_train)
proba = base.predict_proba(X_test)[:, 1]
yt = y_test.to_numpy()
sweep = pd.DataFrame([rates(yt, proba, t) for t in np.round(np.arange(0.05, 0.96, 0.05), 2)])
sweep["legit_blocked_per_10k"] = (sweep["fpr"] * 10_000).round().astype(int)
sweep["phish_missed_per_10k"] = (sweep["fnr"] * 10_000).round().astype(int)
print(sweep[["threshold", "fpr", "fnr", "accuracy", "f1"]].round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(sweep.threshold, sweep.fpr * 100, "o-", color="#4C72B0", label="Legitimate sites blocked (FPR)")
ax.plot(sweep.threshold, sweep.fnr * 100, "s-", color="#DD5555", label="Phishing sites missed (FNR)")
ax.axvline(0.5, ls="--", color="grey", lw=1)
ax.text(0.51, ax.get_ylim()[1] * 0.9, "default 0.5", color="grey", fontsize=8)
ax.set(title="Effect of decision threshold (Logistic Regression)", xlabel="threshold on P(phishing)", ylabel="% of class")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "threshold_tradeoff.png", dpi=150)

# ---- Part B: biased training data -----------------------------------------
GROUP = "age_of_domain"  # -1 = young domain (< 6 months)
young_test = (X_test[GROUP] == -1).to_numpy()
legit_young_train = (X_train[GROUP] == -1) & (y_train == 0)

# Biased sample: keep only 5% of legitimate young domains (e.g. data scraped from
# blocklists + popular, long-established sites). Everything else unchanged.
rng = np.random.default_rng(RANDOM_STATE)
drop = legit_young_train & (rng.random(len(X_train)) > 0.05)
Xb, yb = X_train[~drop], y_train[~drop]
print(f"\nLegit young domains in training: fair={int(legit_young_train.sum())}  biased={int((legit_young_train & ~drop).sum())}")


def group_report(name, proba, t=0.5):
    """Overall metrics plus FPR for legitimate young vs established domains."""
    pred = (proba >= t).astype(int)
    legit = yt == 0
    r = {
        "scenario": name,
        "accuracy": float(accuracy_score(yt, pred)),
        "f1": float(f1_score(yt, pred)),
        "fpr_young": float(pred[legit & young_test].mean()),
        "fpr_established": float(pred[legit & ~young_test].mean()),
        "fnr_overall": float(1 - pred[yt == 1].mean()),
    }
    print(f"{name:<34} acc={r['accuracy']:.4f} f1={r['f1']:.4f} "
          f"FPR young={r['fpr_young']:.3f} established={r['fpr_established']:.3f} FNR={r['fnr_overall']:.3f}")
    return r


biased = model().fit(Xb, yb)
proba_b = biased.predict_proba(X_test)[:, 1]

# Mitigation 1: reweight - give the few remaining legit young-domain samples a weight
# that restores their original share (a standard re-sampling / re-weighting fix).
w = np.ones(len(Xb))
mask = ((Xb[GROUP] == -1) & (yb == 0)).to_numpy()
w[mask] = legit_young_train.sum() / mask.sum()
reweighted = model().fit(Xb, yb, clf__sample_weight=w)

# Mitigation 2: group-specific threshold - stricter evidence needed to block a young domain.
# Pick the lowest threshold that brings young-domain FPR back to the fair-model level.
fair_young_fpr = float((proba[(yt == 0) & young_test] >= 0.5).mean())
t_young = next(t for t in np.arange(0.5, 1.0, 0.01)
               if (proba_b[(yt == 0) & young_test] >= t).mean() <= fair_young_fpr + 1e-9)
adj = proba_b.copy()
adj_pred_young = (proba_b >= t_young)
adj[young_test] = np.where(adj_pred_young[young_test], 1.0, 0.0)  # apply group threshold
adj[~young_test] = (proba_b[~young_test] >= 0.5).astype(float)

scenarios = [
    group_report("Fair training data", proba),
    group_report("Biased training data", proba_b),
    group_report("Biased + reweighting", reweighted.predict_proba(X_test)[:, 1]),
    group_report(f"Biased + young-domain threshold {t_young:.2f}", adj),
]
bias_df = pd.DataFrame(scenarios)

fig, ax = plt.subplots(figsize=(8, 4.2))
x = np.arange(len(bias_df))
ax.bar(x - 0.2, bias_df.fpr_young * 100, 0.4, color="#DD5555", label="Young legit domains blocked")
ax.bar(x + 0.2, bias_df.fpr_established * 100, 0.4, color="#4C72B0", label="Established legit domains blocked")
for i, r in bias_df.iterrows():
    ax.text(i - 0.2, r.fpr_young * 100 + 0.5, f"{r.fpr_young*100:.1f}%", ha="center", fontsize=8)
    ax.text(i + 0.2, r.fpr_established * 100 + 0.5, f"{r.fpr_established*100:.1f}%", ha="center", fontsize=8)
    ax.text(i, -4.5, f"acc {r.accuracy*100:.1f}%", ha="center", fontsize=8, color="dimgrey")
ax.set_xticks(x, ["Fair data", "Biased data", "Biased +\nreweighting", "Biased +\ngroup threshold"])
ax.set(title="False-positive rate by group: biased data hurts new legitimate domains", ylabel="FPR (%)")
ax.set_ylim(-6, max(bias_df.fpr_young.max() * 100 + 6, 10))
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "bias_simulation.png", dpi=150)

save_result("ethics", {
    "threshold_sweep": sweep.round(4).to_dict(orient="records"),
    "bias": bias_df.round(4).to_dict(orient="records"),
    "legit_young_train_fair": int(legit_young_train.sum()),
    "legit_young_train_biased": int((legit_young_train & ~drop).sum()),
    "young_legit_test": int(((yt == 0) & young_test).sum()),
    "t_young": round(float(t_young), 2),
})
