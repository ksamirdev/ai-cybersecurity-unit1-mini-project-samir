"""Builds report/Technical_Report.pdf from outputs/results.json and the saved plots.

Run after the analysis scripts so every number in the report matches the code output.
"""
import json

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from common import OUT, ROOT, RESULTS_FILE

STUDENT = "Samir Kumar Gupta"
ENROLLMENT = "2301730243"
REPO = "https://github.com/ksamirdev/ai-cybersecurity-unit1-mini-project-samir"

R = json.loads(RESULTS_FILE.read_text())
EDA, CMP, ETH = R["eda"], R["comparison"], R["ethics"]
LR, DT, NN = CMP["Logistic Regression"], CMP["Decision Tree"], CMP["Neural Network"]

ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["BodyText"], fontSize=10, leading=14.5, spaceAfter=6)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=15, textColor=colors.HexColor("#1F3A5F"), spaceBefore=10, keepWithNext=1)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, textColor=colors.HexColor("#1F3A5F"), spaceBefore=8, keepWithNext=1)
CAP = ParagraphStyle("cap", parent=BODY, fontSize=8.5, textColor=colors.grey, alignment=TA_CENTER, spaceAfter=10)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=9, leading=12, spaceAfter=0)
fig_no = [0]


def p(text):
    return Paragraph(text, BODY)


BULLET = ParagraphStyle("bullet", parent=BODY, leftIndent=14, bulletIndent=4, spaceAfter=4)


def bullets(items):
    """Individual paragraphs (not a ListFlowable) so headings' keepWithNext still works."""
    return [Paragraph(i, BULLET, bulletText="\u2022") for i in items]


def figure(name, caption, width=15, keep=True):
    """Image + numbered caption. keep=False returns a plain list for use inside table cells."""
    from PIL import Image as PILImage
    path = OUT / name
    w, h = PILImage.open(path).size
    fig_no[0] += 1
    parts = [Image(str(path), width=width * cm, height=width * cm * h / w),
             Paragraph(f"Figure {fig_no[0]}: {caption}", CAP)]
    return KeepTogether(parts) if keep else parts


def table(rows, widths, header=True):
    rows = [[c if isinstance(c, Paragraph) else Paragraph(str(c), CELL) for c in r] for r in rows]
    t = Table(rows, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B0B8C4")),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F6FA")]),
             ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3A5F")),
                  ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
        rows[0] = [Paragraph(f"<b><font color='white'>{c.text}</font></b>", CELL) for c in rows[0]]
        t = Table(rows, colWidths=[w * cm for w in widths], repeatRows=1)
    t.setStyle(TableStyle(style))
    return t


def pct(x):
    return f"{x * 100:.2f}%"


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(2 * cm, 1.2 * cm, f"{STUDENT} | {ENROLLMENT} | ENSP355 Lab Assignment 01")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


s = []

# ---- Cover ---------------------------------------------------------------
title = ParagraphStyle("t", parent=H1, fontSize=22, alignment=TA_CENTER, leading=28)
sub = ParagraphStyle("s", parent=BODY, fontSize=12, alignment=TA_CENTER, leading=17)
s += [Spacer(1, 3 * cm), Paragraph("Lab Assignment 01 - Technical Report", sub), Spacer(1, 0.4 * cm),
      Paragraph("Phishing Website Detection with Machine Learning", title),
      Paragraph("An AI-Driven Cyber Threat Awareness and Detection Prototype", sub), Spacer(1, 1.6 * cm)]
s.append(table([
    ["Course", "AI in Cyber Security Lab"],
    ["Course Code", "ENSP355"],
    ["Submitted By", STUDENT],
    ["Enrollment No.", ENROLLMENT],
    ["Section", "SEC-D"],
    ["Faculty", "Mrs. Monika"],
    ["Assignment Type", "Individual mini project (Unit 1)"],
    ["Repository", f"<link href='{REPO}' color='blue'>{REPO.replace('https://', '')}</link>"],
], [4.5, 10.5], header=False))
s.append(PageBreak())

# ---- 1. Problem overview ---------------------------------------------------
s += [Paragraph("1. Problem Overview", H1),
      p("Phishing is still one of the most common ways attackers get into an organisation. "
        "A fake login page or a lookalike domain collects passwords, card numbers or session tokens, "
        "and blocklists only catch a site after someone has reported it. "
        "New phishing domains appear every day, often live for only a few hours, so a fixed signature list is always behind."),
      p("This project plays the role of an entry-level security analyst who has to answer a practical question: "
        "<i>can a model learn, from simple URL, page and domain properties, whether a website is a phishing site?</i> "
        "The prototype covers the full pipeline asked for in the brief: environment setup, threat-landscape analysis, "
        "a classic ML classifier, a small neural network, and an ethics simulation that shows what goes wrong when "
        "thresholds or training data are chosen carelessly."),
      Paragraph("2. Dataset", H1),
      p("I used the <b>UCI Phishing Websites dataset</b> (Mohammad, Thabtah and McCluskey, 2015), downloaded from OpenML "
        "(data id 4534). It is a real, public dataset collected from PhishTank, MillerSmiles and Google search results, "
        "and it contains no personal data. Each row is one website described by 30 indicators, and the label says "
        "whether the site was phishing or legitimate."),
      p("Every indicator is already encoded by the dataset authors as <b>-1 (phishy)</b>, <b>0 (suspicious)</b> or "
        "<b>1 (legitimate-looking)</b>. Examples:"),
      table([
          ["Group", "Example indicators", "What it captures"],
          ["Address bar", "having_IP_Address, URL_Length, Prefix_Suffix, having_Sub_Domain, SSLfinal_State",
           "IP instead of a name, very long URLs, '-' in the domain, many sub-domains, missing or untrusted HTTPS"],
          ["Page content", "URL_of_Anchor, Request_URL, Links_in_tags, SFH, Submitting_to_email",
           "Links and forms that point to other domains or to email - typical of copied login pages"],
          ["Page behaviour", "on_mouseover, RightClick, popUpWidnow, Iframe, Redirect",
           "Scripts that hide the real URL, disable right-click, or open pop-ups"],
          ["Domain reputation", "age_of_domain, DNSRecord, web_traffic, Page_Rank, Google_Index",
           "Young domains, no DNS record, little traffic, not indexed by Google"],
      ], [3, 6, 6]),
      Spacer(1, 6),
      p(f"<b>Data quality finding.</b> The raw file has {EDA['raw_rows']:,} rows, but <b>{EDA['duplicates']:,} of them "
        f"({EDA['duplicates'] / EDA['raw_rows'] * 100:.0f}%) are exact duplicates</b>. Because the features are coarse "
        "three-level codes, many websites end up with identical rows. If duplicates are kept, the same row can sit in both "
        "the training and the test set, and the test score measures memory instead of detection. I removed them, leaving "
        f"<b>{EDA['rows']:,} unique websites</b> ({EDA['legitimate']:,} legitimate, {EDA['phishing']:,} phishing). "
        f"There are no missing values. {EDA['conflicting']} rows share identical features but carry different labels; "
        "they were kept because they reflect real label noise that no model can resolve.")]

# ---- 3. Methodology -------------------------------------------------------
s += [Paragraph("3. Methodology", H1),
      bullets([
          "<b>Environment (Task 1):</b> Python 3.12 virtual environment, packages pinned in requirements.txt, "
          "validated by <font face='Courier'>src/check_setup.py</font>.",
          "<b>EDA (Task 2):</b> class balance, indicator strength (correlation with the label and phishing rate when an "
          "indicator is set), value frequencies per class, and the number of phishy signals per website.",
          "<b>Preprocessing (Task 3):</b> scikit-learn <font face='Courier'>ColumnTransformer</font>. "
          "Two-level indicators are treated as numeric and standardised with StandardScaler. "
          "Three-level indicators are categorical and one-hot encoded (the 0 'suspicious' level is not halfway between "
          "phishy and legitimate, so it should not be treated as a number). Median / most-frequent imputers are part of the "
          "pipeline so new data with gaps still works. The preprocessor is fitted on training data only.",
          "<b>Split:</b> stratified 80/20 train-test split (random_state = 42), plus 5-fold stratified cross-validation "
          "to check that the comparison is not an accident of one split.",
          "<b>Models:</b> Logistic Regression and a depth-limited Decision Tree (Task 3), and a two-layer MLPClassifier "
          "with early stopping (Task 4). All three use the same pipeline and the same split.",
          "<b>Ethics (Task 5):</b> threshold sweep for false positives, and a biased-training-data experiment on young "
          "domains with two mitigations.",
      ])]

# ---- 4. EDA ----------------------------------------------------------------
s += [Paragraph("4. Task 2 - Threat Landscape Analysis", H1),
      figure("class_distribution.png", "Legitimate vs phishing websites after removing duplicates", 10),
      p(f"The classes are roughly balanced ({EDA['phishing'] / EDA['rows'] * 100:.1f}% phishing), so accuracy is a "
        "meaningful metric here and no resampling is needed. Precision and recall are still reported, because in "
        "security the two kinds of error have very different costs."),
      figure("top_indicators.png", "Indicators most strongly correlated with the phishing label", 11)]
ind_rows = [["Indicator", "|r|", "Sites flagged (-1)", "Phishing rate if flagged", "Phishing rate otherwise"]]
ind_rows += [[i["feature"], i["corr"], f"{i['flagged_share']}%", f"{i['phishing_rate_when_flagged']}%",
              f"{i['phishing_rate_otherwise']}%"] for i in EDA["indicators"]]
s += [Paragraph("Common threat indicators", H2), table(ind_rows, [4.2, 1.6, 3, 3.3, 3.3]), Spacer(1, 8),
      figure("indicator_frequency.png", "Frequency of each value of the six strongest indicators, split by class", 16),
      figure("signal_count_distribution.png", "How many of the 30 indicators are set to 'phishy' per website", 13),
      figure("correlation_heatmap.png", "Correlation between the strongest indicators and the label", 9.5),
      Paragraph("EDA observations", H2),
      bullets([
          "<b>HTTPS state and anchor links dominate.</b> SSLfinal_State and URL_of_Anchor have correlations near 0.7. "
          f"When URL_of_Anchor is phishy (most links on the page point elsewhere), "
          f"{EDA['indicators'][1]['phishing_rate_when_flagged']}% of sites are phishing. "
          "This makes sense: a cloned login page keeps the look of the original but its links still point to the real site.",
          "<b>The 'suspicious' (0) level matters.</b> For SSLfinal_State, the 0 value (HTTPS with an untrusted issuer or "
          "a short-lived certificate) appears almost only on phishing sites. Treating it as its own category is the right choice.",
          f"<b>No single indicator is enough.</b> Phishing sites carry {EDA['mean_signals_phish']} phishy signals on "
          f"average versus {EDA['mean_signals_legit']} for legitimate sites, but the two distributions overlap heavily "
          "(Figure 4). Prefix_Suffix is set on most legitimate sites too. A model has to combine signals.",
          "<b>Domain reputation is weaker than expected.</b> web_traffic and age_of_domain help, but many legitimate "
          "sites are also young or low-traffic. This is exactly the group studied in the ethics section.",
      ])]

# ---- 5. ML ---------------------------------------------------------------------
def metric_rows(models):
    rows = [["Metric"] + list(models)]
    for k, label in [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score")]:
        rows.append([label] + [pct(CMP[m][k]) for m in models])
    rows.append(["5-fold CV F1 (mean +/- std)"] + [f"{CMP[m]['cv_f1_mean'] * 100:.2f}% +/- {CMP[m]['cv_f1_std'] * 100:.2f}"
                                                   for m in models])
    rows.append(["False positives / false negatives"] + [f"{CMP[m]['fp']} / {CMP[m]['fn']}" for m in models])
    rows.append(["Training time"] + [f"{CMP[m]['train_seconds']:.3f} s" for m in models])
    return rows


s += [Paragraph("5. Task 3 - Machine Learning Threat Classification", H1),
      p("Logistic Regression is the baseline: it is fast, and its weights can be read directly. "
        "The Decision Tree is limited to depth 10 with at least 5 samples per leaf so that it learns general rules "
        "rather than memorising individual websites."),
      table(metric_rows(["Logistic Regression", "Decision Tree"]), [6, 4.5, 4.5]), Spacer(1, 8),
      p(f"Both models detect phishing well on unseen data. Logistic Regression reaches {pct(LR['accuracy'])} accuracy and "
        f"{pct(LR['f1'])} F1 on the {LR['tn'] + LR['fp'] + LR['fn'] + LR['tp']} test websites. It missed {LR['fn']} phishing "
        f"sites (false negatives) and wrongly flagged {LR['fp']} legitimate ones (false positives). The cross-validation "
        "scores are close to the single-split scores with a small standard deviation, so the result is stable."),
      Table([[figure("cm_logistic_regression.png", "Logistic Regression confusion matrix", 7.5, keep=False),
              figure("cm_decision_tree.png", "Decision Tree confusion matrix", 7.5, keep=False)]], colWidths=[8.2 * cm] * 2),
      figure("lr_coefficients.png", "Largest Logistic Regression weights (red pushes towards phishing)", 12),
      p("The weights confirm the EDA: phishy anchor links (URL_of_Anchor = -1) and an untrusted HTTPS certificate "
        "(SSLfinal_State = 0) push strongly towards phishing, while a trusted certificate (SSLfinal_State = 1) pushes "
        "towards legitimate. An analyst can explain a Logistic Regression alert in these terms.")]

# ---- 6. DL ---------------------------------------------------------------------
s += [Paragraph("6. Task 4 - Neural Network Prototype", H1),
      p("The neural network is a scikit-learn MLPClassifier:"),
      bullets([
          "Input: the same preprocessed features (standardised binary indicators + one-hot three-level indicators).",
          "Hidden layers: 64 and 32 neurons, ReLU activation.",
          "Optimiser: Adam, L2 penalty alpha = 0.0001.",
          f"Early stopping: 10% of training data held out; training stopped after {NN['epochs']} epochs when the "
          f"validation score stopped improving. The network has {NN['parameters']:,} trainable parameters.",
      ]),
      figure("mlp_training_curve.png", "MLP training loss and validation accuracy per epoch", 10.5),
      figure("cm_neural_network.png", "Neural network confusion matrix", 7.5),
      Paragraph("ML vs DL comparison", H2),
      table(metric_rows(["Logistic Regression", "Decision Tree", "Neural Network"]), [5.4, 3.2, 3.2, 3.2]),
      Spacer(1, 8),
      figure("model_comparison.png", "Detection metrics and training time of all three models", 16),
      Paragraph("Analysis", H2),
      bullets([
          f"<b>Detection accuracy:</b> the MLP is best on every split, but only by a small margin: "
          f"{pct(NN['accuracy'])} vs {pct(LR['accuracy'])} accuracy, and "
          f"{(NN['cv_f1_mean'] - LR['cv_f1_mean']) * 100:.2f} percentage points of cross-validated F1. "
          f"Its main gain is precision: it raised {NN['fp']} false alarms against {LR['fp']} for Logistic Regression. "
          "Recall is about the same.",
          f"<b>Training time:</b> the MLP took {NN['train_seconds']:.2f} s, about "
          f"{NN['train_seconds'] / LR['train_seconds']:.0f}x longer than Logistic Regression. All models predict 1,170 "
          "websites in a few milliseconds, so prediction speed is not a problem for any of them.",
          "<b>Why the gap is small:</b> the inputs are 30 hand-made, already-coarse indicators. Most of the useful "
          "information is in a few strong features that a linear model already uses well, so there are few complex "
          "interactions left for a neural network to find. Deep learning pays off more on raw inputs such as the full URL "
          "text or page HTML.",
          "<b>Practical feasibility:</b> for this data, Logistic Regression is the sensible production choice: almost the "
          "same detection, much cheaper to train and retrain daily, and every alert can be explained by its weights. "
          "The MLP is worth it only if its fewer false alarms matter more than explainability, for example in an "
          "email gateway that blocks links automatically.",
      ])]

# ---- 7. Ethics ----------------------------------------------------------------
sw = {round(r["threshold"], 2): r for r in ETH["threshold_sweep"]}
fair, biased = ETH["bias"][0], ETH["bias"][1]
rew, thr = ETH["bias"][2], ETH["bias"][3]
s += [Paragraph("7. Task 5 - Ethical Reflection and AI Challenges", H1),
      Paragraph("7.1 Simulation A: false positives block legitimate users", H2),
      p("A phishing filter outputs a probability, and someone has to choose the threshold at which a site is blocked. "
        "I trained Logistic Regression and moved the threshold from 0.05 to 0.95. A false positive here is a legitimate "
        "website (a shop, a bank, a university portal) that users cannot open."),
      figure("threshold_tradeoff.png", "Share of legitimate sites blocked and phishing sites missed at each threshold", 12.5),
      table([["Threshold", "Legit sites blocked", "Phishing missed", "Legit blocked per 10,000", "Accuracy"]] +
            [[f"{t:.2f}", pct(sw[t]["fpr"]), pct(sw[t]["fnr"]), f"{sw[t]['legit_blocked_per_10k']:,}", pct(sw[t]["accuracy"])]
             for t in (0.1, 0.3, 0.5, 0.7, 0.9)], [2.5, 3.2, 3.2, 3.6, 2.5]),
      Spacer(1, 8),
      p(f"A security team that wants to 'miss nothing' might pick a low threshold. At 0.10 the filter misses only "
        f"{pct(sw[0.1]['fnr'])} of phishing sites, but it blocks {pct(sw[0.1]['fpr'])} of legitimate ones: about "
        f"{sw[0.1]['legit_blocked_per_10k']:,} out of every 10,000 real websites. At 0.90 the opposite happens: almost no "
        f"legitimate site is blocked, but {pct(sw[0.9]['fnr'])} of phishing gets through. Overall accuracy changes much less "
        "than either error rate, so <b>a single accuracy number hides who pays for the errors</b>. The threshold is a policy "
        "decision about whose harm is acceptable, not only a technical setting."),
      Paragraph("7.2 Simulation B: biased data leads to unfair decisions", H2),
      p("Young domains (registered less than six months ago) are common for phishing, but they are also how every new small "
        "business, startup or event website starts. Imagine the training data was built from a blocklist plus a list of "
        "popular, long-established sites: legitimate young domains would barely appear. To simulate this I kept only 5% of the "
        f"legitimate young-domain websites in the training set ({ETH['legit_young_train_fair']:,} reduced to "
        f"{ETH['legit_young_train_biased']}), changed nothing else, and evaluated on the same untouched test set."),
      figure("bias_simulation.png", "False-positive rate for young vs established legitimate domains under each scenario", 13.5),
      table([["Scenario", "Accuracy", "F1", "Young legit blocked", "Established legit blocked", "Phishing missed"]] +
            [[r["scenario"], pct(r["accuracy"]), pct(r["f1"]), pct(r["fpr_young"]), pct(r["fpr_established"]),
              pct(r["fnr_overall"])] for r in ETH["bias"]], [4.6, 2, 2, 2.4, 2.6, 2.2]),
      Spacer(1, 8),
      bullets([
          f"<b>The harm lands on one group.</b> With biased data the model blocks {pct(biased['fpr_young'])} of legitimate young "
          f"domains, almost {biased['fpr_young'] / fair['fpr_young']:.0f} times the fair model's {pct(fair['fpr_young'])}. "
          f"The false-positive rate for established domains stays at {pct(biased['fpr_established'])}. "
          "In other words the model learned 'new website = phishing'.",
          f"<b>It is easy to miss.</b> Accuracy fell from {pct(fair['accuracy'])} to {pct(biased['accuracy'])}, a drop that "
          "could pass as normal variation. Phishing recall even improved, so a team watching only detection rate would think "
          "the new model is better. The problem only shows up when errors are broken down by group.",
          f"<b>Mitigation 1 - reweighting:</b> giving the few remaining legitimate young domains a higher sample weight brings "
          f"their false-positive rate back to {pct(rew['fpr_young'])}.",
          f"<b>Mitigation 2 - a group-specific threshold</b> of {ETH['t_young']:.2f} for young domains restores "
          f"{pct(thr['fpr_young'])}, but the model then misses more phishing ({pct(thr['fnr_overall'])}). "
          "Fixing a biased model after training always trades one error for another; collecting representative data is the "
          "real fix.",
      ]),
      Paragraph("7.3 Ethical risks", H2),
      bullets([
          "<b>Harm from false positives:</b> a blocked legitimate site loses customers and trust, and repeated false alarms "
          "teach users to ignore or bypass security warnings (alert fatigue).",
          "<b>Harm from false negatives:</b> each missed phishing site can lead to stolen credentials and a breach. "
          "Neither error is free, so thresholds need explicit sign-off and review.",
          "<b>Unfair treatment of groups:</b> as shown above, models can penalise new businesses, small organisations, or "
          "websites from regions that use cheaper hosting or certificate providers, without anyone intending it.",
          "<b>Automation bias:</b> analysts may trust the model's verdict without checking. High-impact actions such as "
          "takedown requests should keep a human in the loop and an appeal route for site owners.",
          "<b>Adversarial adaptation:</b> attackers can buy trusted certificates and age domains before an attack. "
          "A model trained on 2015 data degrades as attackers learn which indicators it uses, so it must be retrained "
          "and monitored for drift.",
      ]),
      Paragraph("7.4 Privacy concerns", H2),
      bullets([
          "This dataset only describes websites, but a deployed filter sees every URL every employee visits. That browsing "
          "history reveals health, religion, politics and personal life, and must be minimised, access-controlled and kept "
          "only as long as needed (data minimisation and purpose limitation under laws like India's DPDP Act 2023 and GDPR).",
          "Logs used for retraining should be stripped of user identifiers, and URL query strings (which often contain "
          "emails, tokens or session IDs) should be removed before storage.",
          "Sending URLs to a third-party cloud model shares this data with another company; on-premise inference or "
          "hashing should be considered.",
      ]),
      Paragraph("7.5 Explainability limitations", H2),
      bullets([
          "Logistic Regression is explainable globally: its weights (Figure 8) show which indicators drive decisions, and "
          "each alert can be explained as a sum of contributions.",
          "The MLP has 5,000+ weights spread across two hidden layers; there is no direct way to say why one particular site "
          "was blocked. Post-hoc tools such as SHAP or LIME can approximate an explanation, but they are approximations and "
          "can be unstable.",
          "Even for the simple model, the indicators are pre-encoded by the dataset authors. A site owner told 'your "
          "SFH = -1' cannot easily understand or fix the cause, so explanations must be translated into plain language.",
          "Because the MLP's accuracy gain is under one percentage point, the loss of explainability is hard to justify for "
          "this use case.",
      ])]

# ---- 8. Results, limitations, conclusion --------------------------------------
s += [Paragraph("8. Results and Key Observations", H1),
      bullets([
          f"Removing {EDA['duplicates']:,} duplicate rows was the most important data step; without it test scores would be inflated.",
          "Trusted HTTPS state and where a page's links point are the strongest phishing indicators.",
          f"Logistic Regression: {pct(LR['accuracy'])} accuracy, {pct(LR['f1'])} F1, trained in {LR['train_seconds']:.3f} s.",
          f"Decision Tree: {pct(DT['accuracy'])} accuracy, {pct(DT['f1'])} F1, fastest to train.",
          f"Neural network: {pct(NN['accuracy'])} accuracy, {pct(NN['f1'])} F1, fewest false alarms, but "
          f"~{NN['train_seconds'] / LR['train_seconds']:.0f}x slower to train and hard to explain.",
          "The threshold choice shifts errors between legitimate users and missed attacks far more than it shifts accuracy.",
          f"Biased training data raised false blocks of new legitimate domains from {pct(fair['fpr_young'])} to "
          f"{pct(biased['fpr_young'])} while overall metrics looked acceptable.",
      ]),
      Paragraph("9. Limitations", H1),
      bullets([
          "The dataset dates from 2015 and uses pre-encoded indicators. Phishing techniques have changed (free HTTPS from "
          "Let's Encrypt has weakened the SSL signal, for example), so these scores would not hold on today's traffic.",
          "The 30 indicators are coarse, which causes many duplicate and some contradictory rows.",
          "Only one neural-network architecture was tried, with default-style hyper-parameters and no tuning.",
          "The bias experiment is a simulation of a plausible sampling error, not a measurement of real-world bias.",
      ]),
      Paragraph("10. Conclusion", H1),
      p("The prototype shows that simple, explainable machine learning can detect most phishing websites from URL, page and "
        f"domain indicators: about {LR['accuracy'] * 100:.0f}% accuracy with Logistic Regression and slightly more with a small "
        "neural network. On this tabular data the neural network adds little detection power for a large cost in training time "
        "and explainability, so the linear model is the more practical choice. The ethics experiments show that the decisions "
        "around the model - which threshold to use and which data to train on - can matter as much as the model itself. "
        "A responsible deployment needs group-level error monitoring, representative training data, privacy controls on browsing "
        "logs, and a human review and appeal path for blocked sites."),
      Paragraph("11. References", H1),
      bullets([
          "R. M. Mohammad, F. Thabtah, L. McCluskey, <i>Phishing Websites</i>, UCI Machine Learning Repository, 2015. "
          "https://doi.org/10.24432/C51W2X",
          "OpenML dataset 4534, PhishingWebsites. https://www.openml.org/d/4534",
          "R. M. Mohammad, F. Thabtah, L. McCluskey, 'An assessment of features related to phishing websites using an "
          "automated technique', ICITST 2012.",
          "F. Pedregosa et al., 'Scikit-learn: Machine Learning in Python', JMLR 12, 2011. https://scikit-learn.org",
          "NIST Cybersecurity Framework 2.0, 2024. https://www.nist.gov/cyberframework",
          "Digital Personal Data Protection Act, 2023, Government of India.",
      ]),
      Paragraph("Appendix - How to reproduce", H2),
      p(f"Repository: <link href='{REPO}' color='blue'>{REPO}</link>. Setup and run commands are in README.md; every figure "
        "and number in this report is regenerated by the scripts in <font face='Courier'>src/</font>.")]

s = [f for x in s for f in (x if isinstance(x, list) else [x])]  # bullets() returns lists
(ROOT / "report").mkdir(exist_ok=True)
out = ROOT / "report" / "Technical_Report.pdf"
SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=2 * cm,
                  title="Phishing Website Detection - Lab Assignment 01", author=STUDENT).build(
    s, onFirstPage=lambda c, d: None, onLaterPages=footer)
print("Report written to", out)
