"""Task 2: exploratory threat-landscape analysis of the phishing dataset.

Produces plots in outputs/ and stores summary numbers in outputs/results.json.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from common import OUT, TARGET, load_data, save_result

sns.set_theme(style="whitegrid")
LABELS = {0: "Legitimate", 1: "Phishing"}
PALETTE = {"Legitimate": "#4C72B0", "Phishing": "#DD5555"}

raw = load_data(dedupe=False)
df = load_data()
features = df.drop(columns=TARGET)
label_name = df[TARGET].map(LABELS)

# ---- Basic structure -------------------------------------------------------
print(f"raw {raw.shape} | duplicate rows removed: {int(raw.duplicated().sum())} | analysed: {df.shape}")
print("missing values:", int(df.isna().sum().sum()))
conflicting = int(features.duplicated(keep=False).sum())  # same features, different label
print("rows sharing identical features but different labels:", conflicting)
counts = label_name.value_counts()
print(counts)

# ---- 1. Normal vs suspicious ----------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x=label_name, hue=label_name, order=["Legitimate", "Phishing"], palette=PALETTE, legend=False, ax=ax)
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom")
ax.set(title="Legitimate vs Phishing Websites", xlabel="", ylabel="Number of websites")
fig.tight_layout()
fig.savefig(OUT / "class_distribution.png", dpi=150)

# ---- 2. Which indicators matter most? -------------------------------------
# A value of -1 means "this indicator looks phishy" in the dataset's coding.
# Indicator strength = share of sites showing the phishy value that are really phishing.
corr = features.corrwith(df[TARGET]).abs().sort_values(ascending=False)
top = corr.head(12)
fig, ax = plt.subplots(figsize=(7, 5))
sns.barplot(x=top.values, y=top.index, color="#DD5555", ax=ax)
ax.set(title="Top threat indicators (|correlation| with phishing label)", xlabel="|Pearson r|", ylabel="")
fig.tight_layout()
fig.savefig(OUT / "top_indicators.png", dpi=150)

indicator_rows = []
for col in top.index[:8]:
    flagged = features[col] == -1
    indicator_rows.append({
        "feature": col,
        "corr": round(float(corr[col]), 3),
        "flagged_share": round(float(flagged.mean()) * 100, 1),
        "phishing_rate_when_flagged": round(float(df.loc[flagged, TARGET].mean()) * 100, 1),
        "phishing_rate_otherwise": round(float(df.loc[~flagged, TARGET].mean()) * 100, 1),
    })

# ---- 3. Frequency of suspicious activity per key indicator -----------------
key = list(top.index[:6])
value_names = {-1: "phishy (-1)", 0: "suspicious (0)", 1: "legit (1)"}
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for ax, col in zip(axes.flat, key):
    sns.countplot(x=features[col].map(value_names), hue=label_name, hue_order=["Legitimate", "Phishing"],
                  order=[v for k, v in value_names.items() if k in features[col].unique()], palette=PALETTE, ax=ax)
    ax.set(title=col, xlabel="", ylabel="count")
    ax.legend(fontsize=8)
fig.suptitle("Frequency of indicator values: legitimate vs phishing")
fig.tight_layout()
fig.savefig(OUT / "indicator_frequency.png", dpi=150)

# ---- 4. Distribution of attack-related signals ----------------------------
# Number of indicators per site that carry the phishy value (-1).
signal_count = (features == -1).sum(axis=1)
fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(x=signal_count, hue=label_name, hue_order=["Legitimate", "Phishing"], palette=PALETTE,
             discrete=True, multiple="layer", alpha=0.55, ax=ax)
ax.set(title="Distribution of phishy signals per website", xlabel="Indicators set to -1 (out of 30)", ylabel="Websites")
fig.tight_layout()
fig.savefig(OUT / "signal_count_distribution.png", dpi=150)

# ---- 5. Correlation between the strongest indicators ----------------------
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(df[list(top.index[:8]) + [TARGET]].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation of strongest indicators with label")
fig.tight_layout()
fig.savefig(OUT / "correlation_heatmap.png", dpi=150)

save_result("eda", {
    "raw_rows": int(raw.shape[0]),
    "raw_legitimate": int((raw[TARGET] == 0).sum()),
    "raw_phishing": int((raw[TARGET] == 1).sum()),
    "conflicting": conflicting,
    "rows": int(df.shape[0]),
    "features": int(features.shape[1]),
    "missing": int(df.isna().sum().sum()),
    "duplicates": int(raw.duplicated().sum()),
    "legitimate": int(counts["Legitimate"]),
    "phishing": int(counts["Phishing"]),
    "mean_signals_legit": round(float(signal_count[df[TARGET] == 0].mean()), 2),
    "mean_signals_phish": round(float(signal_count[df[TARGET] == 1].mean()), 2),
    "indicators": indicator_rows,
})
print("Top indicators:\n", top)
print("Plots saved to", OUT)
