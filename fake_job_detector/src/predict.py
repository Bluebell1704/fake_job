"""
predict.py
==========
Loads the saved model + vectorizer and predicts for ONE job post.

Test from the terminal (runs the 5 demo ads):
    python -m src.predict
"""
import json
import re
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.preprocessing import (FREE_EMAIL_RE, PHONE_RE, STRUCT_COLUMNS,  # noqa: E402
                               SUSPICIOUS_PHRASES, URGENT_WORDS, caps_ratio,
                               prepare_dataframe, to_feature_matrix)

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MAX_FIELD_CHARS = 20000          # ignore absurdly long input
MIN_TEXT_CHARS = 30              # need at least this much text to say anything useful

# e.g. "$5000 per week" / "₹2,000 daily"
PAY_CLAIM_RE = re.compile(r"[\$₹£€]\s?(\d[\d,]*)\s*(?:/|per|a|each)?\s*(day|daily|week|weekly)")


class JobPostDetector:
    """Wraps the saved files so the app only calls detector.predict(...)."""

    def __init__(self, models_dir=MODELS_DIR):
        models_dir = Path(models_dir)
        needed = ["model.pkl", "tfidf_vectorizer.pkl", "top_terms.json"]
        missing = [f for f in needed if not (models_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing {missing} in {models_dir}. Train the model first:  python -m src.train_model")
        self.model = joblib.load(models_dir / "model.pkl")
        self.vectorizer = joblib.load(models_dir / "tfidf_vectorizer.pkl")
        with open(models_dir / "top_terms.json", encoding="utf-8") as f:
            terms = json.load(f)
        self.fake_terms = {t: w for t, w in terms["fake"] if w > 0}
        self.real_terms = {t: w for t, w in terms["real"] if w < 0}

    # ------------------------------------------------------------------
    def _to_frame(self, fields):
        """Turn the app's input dictionary into a one-row DataFrame with dataset column names."""
        if not isinstance(fields, dict):
            raise ValueError("Input must be a dictionary of job fields.")

        def get(key):
            value = fields.get(key, "")
            return "" if value is None else str(value).strip()[:MAX_FIELD_CHARS]

        description = get("description")
        full_text = get("full_text")               # a whole pasted advertisement
        if full_text:
            description = (description + "\n" + full_text).strip()

        key_text = " ".join([get("title"), get("company_profile"), description,
                             get("requirements"), get("benefits")])
        if len(key_text.strip()) < MIN_TEXT_CHARS:
            raise ValueError("Please enter more text (a title and description, or paste the full advertisement).")

        row = {
            "title": get("title"), "location": get("location"),
            "salary_range": get("salary_range"), "company_profile": get("company_profile"),
            "description": description, "requirements": get("requirements"),
            "benefits": get("benefits"), "employment_type": get("employment_type"),
            "required_experience": get("experience"), "required_education": get("education"),
            "industry": get("industry"), "function": get("function"),
            "telecommuting": int(bool(fields.get("telecommuting", False))),
            "has_company_logo": int(bool(fields.get("has_logo", False))),
            "has_questions": int(bool(fields.get("has_questions", False))),
        }
        return pd.DataFrame([row])

    # ------------------------------------------------------------------
    def predict(self, fields):
        """Return a dictionary with the label, probabilities and explanation."""
        df = prepare_dataframe(self._to_frame(fields))
        X = to_feature_matrix(df, self.vectorizer)
        p_fake = float(self.model.predict_proba(X)[0, 1])
        is_fake = p_fake >= 0.5
        row = df.iloc[0]
        indicators, positives = find_indicators(row)

        # words of this ad that are strongly linked to fake / real ads in the training data
        words = row["clean_text"].split()
        grams = set(words) | {" ".join(p) for p in zip(words, words[1:])}
        fake_hits = sorted(((t, self.fake_terms[t]) for t in grams if t in self.fake_terms),
                           key=lambda x: -x[1])[:8]
        real_hits = sorted(((t, self.real_terms[t]) for t in grams if t in self.real_terms),
                           key=lambda x: x[1])[:8]
        return {
            "label": "POTENTIALLY FAKE JOB POST" if is_fake else "REAL JOB POST",
            "is_fake": is_fake,
            "fake_probability": p_fake,
            "confidence": p_fake if is_fake else 1 - p_fake,
            "indicators": indicators,
            "positive_signals": positives,
            "fake_terms": fake_hits,
            "real_terms": real_hits,
            "clean_text": row["clean_text"],
        }


# ----------------------------------------------------------------------
# Human-readable indicators (rule-based, independent of the ML model).
# They are WARNING SIGNS, not proof of fraud.
# ----------------------------------------------------------------------
def find_indicators(row):
    """Return (warnings, positive_signals) as lists of (category, detail) tuples."""
    text = row["raw_text"].lower()
    desc = row["description"]
    warnings, positives = [], []

    found = [p for p in SUSPICIOUS_PHRASES if p in text]
    if found:
        warnings.append(("Suspicious wording", "phrases such as: " + ", ".join(f'"{p}"' for p in found[:5])))

    profile = row["company_profile"].strip()
    if not profile:
        warnings.append(("Missing company information", "no company profile was provided"))
    elif len(profile) < 50:
        warnings.append(("Missing company information", "the company profile is very short"))
    else:
        positives.append(("Company information", "a company profile is present"))

    for amount, period in PAY_CLAIM_RE.findall(text):
        try:
            if float(amount.replace(",", "")) >= 1000:
                warnings.append(("Unrealistic salary claim",
                                 f"promises {amount} per {period} - unusually high for a daily/weekly wage"))
                break
        except ValueError:
            pass

    urgent = sum(text.count(w) for w in URGENT_WORDS)
    exclaim = row["raw_text"].count("!")
    if urgent >= 2 or exclaim >= 3:
        warnings.append(("Excessive urgency", f"{urgent} urgent phrase(s) and {exclaim} exclamation mark(s)"))

    if not row["requirements"].strip():
        warnings.append(("Missing requirements", "no qualifications or requirements are listed"))
    else:
        positives.append(("Requirements", "qualifications/requirements are listed"))

    contact = []
    if FREE_EMAIL_RE.search(text):
        contact.append("a free e-mail address (Gmail/Yahoo/...) instead of a company domain")
    if "whatsapp" in text or "telegram" in text:
        contact.append("contact through WhatsApp/Telegram")
    if PHONE_RE.search(desc.lower()):
        contact.append("a phone number inside the description")
    if contact:
        warnings.append(("Suspicious contact information", "; ".join(contact)))

    unusual = []
    if len(desc) < 150:
        unusual.append("the description is extremely short")
    if caps_ratio(row["title"] + " " + desc) > 0.30:
        unusual.append("a large share of the text is in CAPITAL LETTERS")
    if not row["location"].strip():
        unusual.append("no location is given")
    if unusual:
        warnings.append(("Unusual description pattern", "; ".join(unusual)))
    elif len(desc) >= 500:
        positives.append(("Description", "a detailed description is provided"))

    return warnings, positives


# ----------------------------------------------------------------------
if __name__ == "__main__":
    from src.sample_jobs import SAMPLE_JOBS

    detector = JobPostDetector()
    for job in SAMPLE_JOBS:
        res = detector.predict(job)
        print("-" * 70)
        print(job["name"], f"   [human expectation: {job['expected']}]")
        print(f"  Prediction : {res['label']}")
        print(f"  Confidence : {min(res['confidence'], 0.99) * 100:.1f}%   (fake score {min(res['fake_probability'], 0.99) * 100:.1f}%)")
        for cat, detail in res["indicators"]:
            print(f"  ! {cat}: {detail}")
    print("-" * 70)
    print("Reminder: these are demo ads and the output is a screening aid, not proof.")
