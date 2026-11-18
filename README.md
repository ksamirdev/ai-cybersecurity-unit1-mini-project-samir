# AI-Driven Phishing Website Detection Prototype

ENSP355 AI in Cyber Security Lab - Lab Assignment 01 (Unit 1 mini project).
Samir Kumar Gupta, 2301730243.

This project detects phishing websites from URL, page and domain indicators.
It covers EDA, classic ML (Logistic Regression, Decision Tree), a neural network (MLPClassifier), and an ethics simulation of false positives and biased training data.
The full write-up is in [`report/Technical_Report.pdf`](report/Technical_Report.pdf).

## Dataset

UCI Phishing Websites dataset (Mohammad, Thabtah & McCluskey, 2015), taken from [OpenML dataset 4534](https://www.openml.org/d/4534).
DOI: [10.24432/C51W2X](https://doi.org/10.24432/C51W2X).
It has 11,055 websites, 30 indicators coded as -1 / 0 / 1, and a phishing / legitimate label.
The file is included at `data/PhishingWebsites.arff`.
The code drops the 5,206 exact duplicate rows, which leaves 5,849 unique websites, so the train and test sets never share a row.

## Project structure

```
data/PhishingWebsites.arff     dataset
src/common.py                  loading, split, preprocessing pipeline, evaluation helper
src/check_setup.py             Task 1 - environment validation
src/threat_analysis.py         Task 2 - EDA and threat-indicator plots
src/ml_classifier.py           Task 3 - Logistic Regression + Decision Tree
src/dl_classifier.py           Task 4 - MLP neural network + ML vs DL comparison
src/ethics_simulation.py       Task 5 - threshold and bias simulations
src/build_report.py            builds report/Technical_Report.pdf from the results
outputs/                       generated plots, results.json, model_comparison.csv
report/Technical_Report.pdf    technical report
```

## Setup

Requires Python 3.10-3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/check_setup.py          # prints package versions and "setup OK"
```

## Usage

Run the scripts in this order.
Each one writes its plots to `outputs/` and adds its numbers to `outputs/results.json`.

```bash
python src/threat_analysis.py      # Task 2
python src/ml_classifier.py        # Task 3
python src/dl_classifier.py        # Task 4 (needs Task 3 results)
python src/ethics_simulation.py    # Task 5
python src/build_report.py         # regenerate the PDF report
```

Every run uses `random_state=42`, so the results are reproducible.
Timings vary from machine to machine.

## Results (test set, 1,170 websites)

| Model | Accuracy | Precision | Recall | F1 | 5-fold CV F1 | Train time |
|---|---|---|---|---|---|---|
| Logistic Regression | 93.59% | 94.60% | 92.88% | 93.73% | 93.76% | 0.03 s |
| Decision Tree | 92.56% | 93.89% | 91.56% | 92.71% | 93.69% | 0.01 s |
| Neural Network (MLP 64-32) | 94.19% | 96.05% | 92.55% | 94.27% | 94.06% | 0.2 s |

Ethics simulation: when legitimate young domains are mostly missing from the training data, the model wrongly blocks 36.0% of them, up from 7.4%.
Overall accuracy only drops from 93.6% to 88.7%, so the harm is easy to miss.
Reweighting or a per-group threshold brings that rate back to about 7-8%.

## References

- R. M. Mohammad, F. Thabtah, L. McCluskey. Phishing Websites. UCI Machine Learning Repository, 2015.
- Pedregosa et al. Scikit-learn: Machine Learning in Python. JMLR 12, 2011.
- NIST Cybersecurity Framework 2.0, 2024.
