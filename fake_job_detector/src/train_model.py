"""
train_model.py
==============
Loads the dataset, preprocesses it, trains FOUR models, compares them, picks one
and saves everything the app needs.

Run from the project folder:
    python -m src.train_model
    python -m src.train_model --data data/fake_job_postings.csv
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")                      # draw charts into files (no window needed)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.preprocessing import (STRUCT_COLUMNS, TARGET_COLUMN, TEXT_COLUMNS,  # noqa: E402
                               prepare_dataframe, to_feature_matrix)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "fake_job_postings.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
SEED = 42                                   # fixed number => same split every run (reproducible)


def load_data(path):
    """Read the CSV and make sure the target column is usable."""
    path = Path(path)
    if not path.exists():
        sys.exit(
            f"\nDataset not found: {path}\n"
            "Download 'fake_job_postings.csv' (EMSCAD) and put it in the data/ folder.\n"
            "See README.md, step 'Download the dataset'.\n"
        )
    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        sys.exit(f"The CSV must contain a '{TARGET_COLUMN}' column (1 = fake, 0 = real).")
    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    return df


def get_models():
    """The four candidate models. class_weight='balanced' tells a model that
    mistakes on the rare FAKE class matter more (we have far fewer fake posts)."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=SEED),
        "Naive Bayes": MultinomialNB(alpha=0.1),
        # LinearSVC gives no probabilities, so we wrap it to obtain them.
        "Support Vector Machine": CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", random_state=SEED), cv=3),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced_subsample",
            n_jobs=-1, random_state=SEED),
    }


def evaluate(name, model, X_test, y_test):
    """Compute all metrics for the FAKE class (label 1)."""
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision (fake)": precision_score(y_test, pred, zero_division=0),
        "Recall (fake)": recall_score(y_test, pred, zero_division=0),
        "F1-score (fake)": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, proba),
    }, pred


def choose_model(results):
    """Pick the model with the best F1 on the FAKE class. If several are within
    0.01 of the best, prefer the simpler / more explainable one (list order)."""
    best = results["F1-score (fake)"].max()
    close = results[results["F1-score (fake)"] >= best - 0.01]
    return close.iloc[0]["Model"]          # results are ordered simple -> complex


def save_plots(results, cms, y, reports_dir):
    """Class balance chart, confusion matrices and the model comparison chart."""
    # 1. class balance
    plt.figure(figsize=(4.5, 3.5))
    ax = sns.countplot(x=y.map({0: "Real", 1: "Fake"}), order=["Real", "Fake"],
                       hue=y.map({0: "Real", 1: "Fake"}), palette=["#2e9e5b", "#d64545"], legend=False)
    ax.set_title("Class distribution (whole dataset)")
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(reports_dir / "class_distribution.png", dpi=150)
    plt.close()

    # 2. confusion matrices (one per model)
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for ax, (name, cm) in zip(axes.ravel(), cms.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(reports_dir / "confusion_matrices.png", dpi=150)
    plt.close()

    # 3. model comparison chart
    metrics = ["Accuracy", "Precision (fake)", "Recall (fake)", "F1-score (fake)"]
    results.set_index("Model")[metrics].plot(kind="bar", figsize=(9, 5), ylim=(0, 1.05))
    plt.title("Model comparison (test set)")
    plt.ylabel("Score")
    plt.xticks(rotation=15)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(reports_dir / "model_comparison.png", dpi=150)
    plt.close()


def run_training(data_path=DATA_PATH, models_dir=MODELS_DIR, reports_dir=REPORTS_DIR,
                 test_size=0.2, verbose=True):
    """The whole training pipeline. Returns the comparison table (a DataFrame)."""
    say = print if verbose else (lambda *a, **k: None)
    models_dir, reports_dir = Path(models_dir), Path(reports_dir)
    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    # ---- 1. load + preprocess -------------------------------------------
    df = load_data(data_path)
    say(f"Loaded {len(df)} job posts.")
    say("Class counts (0 = real, 1 = fake):")
    say(df[TARGET_COLUMN].value_counts().to_string())

    say("\nPreprocessing text (this can take a minute on the full dataset)...")
    df = prepare_dataframe(df)

    # Remove exact duplicate ads BEFORE splitting, otherwise the same ad could
    # appear in both train and test (data leakage => fake-good scores).
    before = len(df)
    df = df.drop_duplicates(subset="raw_text").reset_index(drop=True)
    say(f"Removed {before - len(df)} duplicate posts (avoids leakage). {len(df)} remain.")

    # ---- 2. train/test split (80/20, keeps the real:fake ratio, reproducible) ----
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df[TARGET_COLUMN], random_state=SEED)
    y_train, y_test = train_df[TARGET_COLUMN], test_df[TARGET_COLUMN]
    say(f"Train size: {len(train_df)} | Test size: {len(test_df)}")

    # ---- 3. TF-IDF: learn the vocabulary from TRAINING text only (no leakage) ----
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2),
                                 min_df=5, max_df=0.9, sublinear_tf=True)
    vectorizer.fit(train_df["clean_text"])
    X_train = to_feature_matrix(train_df, vectorizer)
    X_test = to_feature_matrix(test_df, vectorizer)
    say(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_)} words/phrases "
        f"(+ {len(STRUCT_COLUMNS)} extra features)")

    # ---- 4. train + evaluate every model --------------------------------
    rows, cms, fitted, reports = [], {}, {}, []
    for name, model in get_models().items():
        say(f"\nTraining {name} ...")
        model.fit(X_train, y_train)
        metrics, pred = evaluate(name, model, X_test, y_test)
        rows.append(metrics)
        cms[name] = confusion_matrix(y_test, pred, labels=[0, 1])
        fitted[name] = model
        reports.append(f"===== {name} =====\n" + classification_report(
            y_test, pred, target_names=["Real", "Fake"], zero_division=0))

    results = pd.DataFrame(rows).round(4)
    say("\n================ MODEL COMPARISON (test set) ================")
    say(results.to_string(index=False))

    # ---- 5. choose the model ------------------------------------------------
    chosen = choose_model(results)
    say(f"\nSelected model: {chosen}")
    say("(Chosen by F1-score on the FAKE class, not by accuracy - accuracy is misleading "
        "when fake posts are rare.)")

    # ---- 6. save everything the app needs -----------------------------------
    joblib.dump(fitted[chosen], models_dir / "model.pkl")
    joblib.dump(vectorizer, models_dir / "tfidf_vectorizer.pkl")

    # Words that pushed Logistic Regression towards FAKE / REAL (used for explanations;
    # this works for whichever model was chosen).
    lr = fitted["Logistic Regression"]
    vocab = np.array(vectorizer.get_feature_names_out())
    coefs = lr.coef_[0][:len(vocab)]
    order = np.argsort(coefs)
    top_terms = {
        "fake": [[str(vocab[i]), float(coefs[i])] for i in order[::-1][:150]],
        "real": [[str(vocab[i]), float(coefs[i])] for i in order[:150]],
    }
    with open(models_dir / "top_terms.json", "w", encoding="utf-8") as f:
        json.dump(top_terms, f)

    chosen_row = results[results["Model"] == chosen].iloc[0].to_dict()
    info = {"selected_model": chosen, "test_metrics": chosen_row,
            "train_size": int(len(train_df)), "test_size": int(len(test_df)),
            "fake_share_in_data": float(df[TARGET_COLUMN].mean()),
            "tfidf_vocabulary_size": int(len(vectorizer.vocabulary_))}
    with open(models_dir / "model_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    results.to_csv(reports_dir / "model_comparison.csv", index=False)
    (reports_dir / "classification_reports.txt").write_text("\n".join(reports), encoding="utf-8")
    save_plots(results, cms, df[TARGET_COLUMN], reports_dir)

    say(f"\nSaved model + vectorizer in {models_dir}")
    say(f"Saved tables and charts in {reports_dir}")
    say("\nClassification report of the selected model:\n")
    say([r for r in reports if r.startswith(f"===== {chosen} =====")][0])
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the fake job post detector.")
    parser.add_argument("--data", default=str(DATA_PATH), help="path to fake_job_postings.csv")
    args = parser.parse_args()
    run_training(args.data)
