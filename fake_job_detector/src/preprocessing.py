"""
preprocessing.py
================
All text-cleaning and feature-building code lives in this ONE file so that
training (train_model.py) and prediction (predict.py / app.py) use exactly the
same steps. If training and prediction cleaned text differently, the model
would see "unfamiliar" data and perform badly.
"""
import re

import nltk
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from scipy.sparse import csr_matrix, hstack

# --------------------------------------------------------------------------
# 1. Column names used by the project (names follow the EMSCAD / Kaggle CSV)
# --------------------------------------------------------------------------
TARGET_COLUMN = "fraudulent"          # 1 = fake job post, 0 = real job post

# Columns that contain words. These are merged into ONE text string per job.
TEXT_COLUMNS = [
    "title", "department", "company_profile", "description", "requirements",
    "benefits", "employment_type", "required_experience", "required_education",
    "industry", "function",
]

# --------------------------------------------------------------------------
# 2. Hand-made "warning sign" vocabulary (heuristics, not proof of fraud)
# --------------------------------------------------------------------------
SUSPICIOUS_PHRASES = [
    "no experience needed", "no experience required", "no experience necessary",
    "no interview", "earn money", "earn up to", "make money", "easy money",
    "guaranteed income", "unlimited income", "be your own boss",
    "get paid daily", "weekly payout", "registration fee", "processing fee",
    "training fee", "security deposit", "pay a fee", "wire transfer",
    "western union", "money order", "bank account", "credit card",
    "send your resume to", "whatsapp", "telegram", "personal details",
]
URGENT_WORDS = [
    "urgent", "urgently", "immediately", "immediate joining", "apply now",
    "hurry", "limited seats", "limited slots", "last chance", "act now",
    "asap", "today only", "don't miss", "dont miss",
]

# --------------------------------------------------------------------------
# 3. Regular expressions
# --------------------------------------------------------------------------
# EMSCAD hides real links/emails/phones as tokens like "#URL_ab12cd#", so we
# recognise both the hidden form and normal links.
URL_RE = re.compile(r"https?://\S+|www\.\S+|#url_[a-z0-9]+#")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|#email_[a-z0-9]+#")
PHONE_RE = re.compile(
    r"#phone_[a-z0-9]+#|(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}"
)
FREE_EMAIL_RE = re.compile(r"@(gmail|yahoo|hotmail|outlook|rediffmail|aol|protonmail)\.")
HTML_RE = re.compile(r"<[^>]+>")


# --------------------------------------------------------------------------
# 4. NLP tools
# --------------------------------------------------------------------------
def _load_stop_words():
    """Load NLTK stop words (downloads them once). Falls back to scikit-learn's list."""
    try:
        from nltk.corpus import stopwords
        try:
            words = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            words = set(stopwords.words("english"))
    except Exception:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        words = set(ENGLISH_STOP_WORDS)
    # "no" and "not" carry meaning in scam text ("NO experience needed"), so keep them.
    return words - {"no", "not", "nor"}


STOP_WORDS = _load_stop_words()
_STEMMER = PorterStemmer()                     # stemming: "earning" -> "earn"
_TOKENIZER = RegexpTokenizer(r"[a-z]+")        # a "token" = a run of letters


# --------------------------------------------------------------------------
# 5. Text cleaning
# --------------------------------------------------------------------------
def _mask_contacts(text):
    """Replace emails / links / phone numbers by tokens (keeps the FACT that they exist)."""
    text = EMAIL_RE.sub(" emailtoken ", text)
    text = URL_RE.sub(" urltoken ", text)
    return PHONE_RE.sub(" phonetoken ", text)


def clean_text(text):
    """Turn one raw text into a cleaned string of stemmed words.

    Steps: missing-value handling -> lowercase -> remove HTML -> mask
    emails/links/phones -> remove punctuation & special characters and
    tokenize -> remove stop words -> stem.
    """
    if pd.isna(text):                                   # 1. missing value -> empty text
        return ""
    text = str(text).lower()                            # 2. lowercase
    text = HTML_RE.sub(" ", text)                       # 3. remove HTML tags
    text = _mask_contacts(text)                         # 4. mask emails / links / phones
    tokens = _TOKENIZER.tokenize(text)                  # 5. tokenize (letters only => punctuation,
    #                                                        digits and special characters disappear)
    tokens = [t for t in tokens if len(t) > 1 and t not in STOP_WORDS]   # 6. stop words
    tokens = [_STEMMER.stem(t) for t in tokens]         # 7. stemming
    return " ".join(tokens)


def explain_steps(text):
    """Return every intermediate stage of preprocessing (used by the Streamlit app to teach)."""
    lowered = str(text).lower()
    masked = _mask_contacts(HTML_RE.sub(" ", lowered))
    tokens = _TOKENIZER.tokenize(masked)
    no_stop = [t for t in tokens if len(t) > 1 and t not in STOP_WORDS]
    stemmed = [_STEMMER.stem(t) for t in no_stop]
    return {
        "1. Original text": str(text),
        "2. Lowercase": lowered,
        "3. After masking contacts, removing punctuation/special characters and tokenizing": tokens,
        "4. After removing stop words": no_stop,
        "5. After stemming (the words given to TF-IDF)": stemmed,
    }


# --------------------------------------------------------------------------
# 6. Missing values + combining fields
# --------------------------------------------------------------------------
def ensure_columns(df):
    """Make sure every expected column exists and has no missing values."""
    df = df.copy()
    for col in TEXT_COLUMNS + ["location", "salary_range"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)          # missing text -> ""
    for col in ["telecommuting", "has_company_logo", "has_questions"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(0, 1).astype(int)
    return df


def combine_text_fields(df):
    """Join all text columns of each row into one long string."""
    combined = pd.Series("", index=df.index)
    for col in TEXT_COLUMNS:
        combined = combined + " " + df[col]
    return combined.str.strip()


# --------------------------------------------------------------------------
# 7. Extra (structured) features that sit next to TF-IDF
#    Every value is scaled to 0..1 so all models (incl. Naive Bayes) can use them.
# --------------------------------------------------------------------------
STRUCT_COLUMNS = [
    "telecommuting", "has_company_logo", "has_questions",
    "missing_company_profile", "missing_requirements", "missing_benefits",
    "missing_location", "missing_salary", "short_description",
    "description_length", "url_count", "email_count", "exclamation_count",
    "caps_ratio", "urgent_count", "suspicious_phrase_count", "salary_size",
]


def salary_max(salary_text):
    """Largest number inside a salary string such as '20000-28000' (0 if none)."""
    numbers = re.findall(r"\d[\d,]*\.?\d*", str(salary_text))
    values = []
    for n in numbers:
        try:
            values.append(float(n.replace(",", "")))
        except ValueError:
            pass
    return max(values) if values else 0.0


def caps_ratio(text):
    """Share of letters that are UPPERCASE (0..1)."""
    letters = [c for c in text if c.isalpha()]
    return sum(c.isupper() for c in letters) / len(letters) if letters else 0.0


def build_structured_features(df):
    """Compute the hand-made features for every row of an ensure_columns() DataFrame."""
    raw = combine_text_fields(df)
    low = raw.str.lower()
    out = pd.DataFrame(index=df.index)
    out["telecommuting"] = df["telecommuting"]
    out["has_company_logo"] = df["has_company_logo"]
    out["has_questions"] = df["has_questions"]
    out["missing_company_profile"] = (df["company_profile"].str.strip() == "").astype(int)
    out["missing_requirements"] = (df["requirements"].str.strip() == "").astype(int)
    out["missing_benefits"] = (df["benefits"].str.strip() == "").astype(int)
    out["missing_location"] = (df["location"].str.strip() == "").astype(int)
    out["missing_salary"] = (df["salary_range"].str.strip() == "").astype(int)
    desc_len = df["description"].str.len()
    out["short_description"] = (desc_len < 200).astype(int)
    out["description_length"] = (desc_len / 5000).clip(0, 1)
    out["url_count"] = (low.apply(lambda t: len(URL_RE.findall(t))) / 5).clip(0, 1)
    out["email_count"] = (low.apply(lambda t: len(EMAIL_RE.findall(t))) / 3).clip(0, 1)
    out["exclamation_count"] = (raw.str.count("!") / 5).clip(0, 1)
    out["caps_ratio"] = ((df["title"] + " " + df["description"]).apply(caps_ratio) / 0.5).clip(0, 1)
    out["urgent_count"] = (low.apply(lambda t: sum(t.count(w) for w in URGENT_WORDS)) / 5).clip(0, 1)
    out["suspicious_phrase_count"] = (
        low.apply(lambda t: sum(t.count(p) for p in SUSPICIOUS_PHRASES)) / 5
    ).clip(0, 1)
    out["salary_size"] = (
        df["salary_range"].apply(salary_max).apply(np.log1p) / np.log1p(1_000_000)
    ).clip(0, 1)
    return out[STRUCT_COLUMNS]


# --------------------------------------------------------------------------
# 8. One function that does everything to a DataFrame
# --------------------------------------------------------------------------
def prepare_dataframe(df):
    """Raw job DataFrame -> DataFrame with 'raw_text', 'clean_text' and all STRUCT_COLUMNS."""
    df = ensure_columns(df)
    df["raw_text"] = combine_text_fields(df)
    df["clean_text"] = df["raw_text"].apply(clean_text)
    struct = build_structured_features(df)
    for col in STRUCT_COLUMNS:
        df[col] = struct[col]
    return df


def to_feature_matrix(df_prepared, vectorizer):
    """TF-IDF numbers for the text + the structured features, side by side."""
    tfidf = vectorizer.transform(df_prepared["clean_text"])
    extra = csr_matrix(df_prepared[STRUCT_COLUMNS].values.astype(float))
    return hstack([tfidf, extra]).tocsr()
