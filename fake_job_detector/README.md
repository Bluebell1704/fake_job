# NLP-Based Fake Job Post Detector

A beginner-friendly academic project that reads a job advertisement, analyses its text with **Natural Language Processing (NLP)** and predicts, with a **machine-learning** model, whether it is a **REAL JOB POST** or a **POTENTIALLY FAKE JOB POST**. The Streamlit app shows a confidence percentage and explains which warning signs influenced the result.

> ⚠️ **Important:** this is a screening / decision-support tool. It learns patterns from historical data and can be wrong. It never proves that a job is a scam (or genuine). Always verify an employer independently, and never pay money or share bank details to get a job.

---

## 1. Project overview (in very simple words)

Fake job ads (paid "registration fees", "earn $5000 a week from home", vague companies…) share writing habits. If we show a computer thousands of ads that humans already labelled *real* or *fake*, it can learn those habits.

1. **Text → numbers.** Computers cannot read words, so we clean the text and turn it into numbers (TF-IDF).
2. **Learn.** Four different algorithms study the numbers of the training ads.
3. **Test.** We keep some ads hidden, let each algorithm predict them, and measure the mistakes.
4. **Use.** The best model is saved. The web app loads it and scores a new ad.

## 2. Architecture

```
Job Posting
   ↓
Data Collection (fake_job_postings.csv)
   ↓
Data Cleaning (missing values, duplicates)
   ↓
Text Preprocessing (lowercase, remove punctuation/special characters)
   ↓
Tokenization
   ↓
Stop-word Removal
   ↓
Stemming (word → root word)
   ↓
TF-IDF Feature Extraction  (+ 17 hand-made features)
   ↓
Machine Learning Model (Logistic Regression / Naive Bayes / SVM / Random Forest → best one saved)
   ↓
Prediction  (REAL / POTENTIALLY FAKE)
   ↓
Confidence Score
   ↓
Explanation (warning indicators + influential words)
   ↓
Streamlit Interface
```

## 3. Dataset

**Where it comes from.** The project is built for the **Employment Scam Aegean Dataset (EMSCAD)**, created by the Laboratory of Information & Communication Systems Security at the University of the Aegean (Greece). A copy is widely shared on **Kaggle** as *"Real / Fake Job Posting Prediction"* with the file name `fake_job_postings.csv`.

**Exact numbers.** Do not trust numbers copied from the internet – your copy may differ. When you train, the script prints the real row count and the real/fake counts. It is commonly reported that the dataset has roughly seventeen to eighteen thousand postings and that only a small percentage (roughly 5%) are fake, i.e. it is **imbalanced**. *Assumption:* the CSV you download has the columns listed below. If a column is missing, the code creates it as empty (no crash).

| Column | Meaning |
|---|---|
| `job_id` | Row number/ID. **Not used** (an ID says nothing about fraud). |
| `title` | Job title. |
| `location` | Country, state, city. |
| `department` | Department of the company. |
| `salary_range` | Salary text such as `20000-28000` (mostly empty). |
| `company_profile` | Short description of the company. |
| `description` | Main text of the advertisement. |
| `requirements` | Skills/qualifications requested. |
| `benefits` | Benefits offered. |
| `telecommuting` | 1 = remote/work-from-home position. |
| `has_company_logo` | 1 = the ad showed a company logo. |
| `has_questions` | 1 = the ad contained screening questions. |
| `employment_type` | Full-time, Part-time, Contract, Temporary, Other. |
| `required_experience` | Entry level, Mid-Senior level, … |
| `required_education` | Bachelor's Degree, High School, … |
| `industry` | Industry of the employer. |
| `function` | Job function (Engineering, Sales, …). |
| **`fraudulent`** | **TARGET variable.** 1 = fake job post, 0 = real job post. |

**Missing values.** Many text columns are empty for many ads.
- Text columns: missing → empty string `""` (so the pipeline never crashes).
- Yes/no columns (`telecommuting`, `has_company_logo`, `has_questions`): missing → `0`.
- Missing information is *itself a clue*, so we also create features such as `missing_company_profile`, `missing_requirements`, `missing_location`, `missing_salary`.
- Rows without a `fraudulent` label are dropped (cannot learn from them).
- Exact duplicate ads are removed before splitting so the same ad can't appear in both training and test data.

**Train/test split.** 80% training, 20% testing, using `train_test_split(..., stratify=fraudulent, random_state=42)`:
- *stratify* keeps the same real:fake ratio in both parts (important because fake ads are rare),
- *random_state=42* makes the split identical each run (reproducible),
- the TF-IDF vocabulary is learned from **training text only**, so nothing from the test set leaks into training.

## 4. Folder structure

```
fake_job_detector/
├── data/
│   └── fake_job_postings.csv     ← you download this (see step 7)
├── models/                       ← created by training
│   ├── model.pkl                 ← the selected trained classifier
│   ├── tfidf_vectorizer.pkl      ← the fitted TF-IDF vectorizer (vocabulary + IDF weights)
│   ├── top_terms.json            ← words most linked to fake / real ads (for explanations)
│   └── model_info.json           ← which model was chosen + its test scores
├── reports/                      ← created by training: tables and charts
│   ├── model_comparison.csv / .png
│   ├── confusion_matrices.png
│   ├── class_distribution.png
│   └── classification_reports.txt
├── notebooks/
│   └── model_training.ipynb      ← step-by-step notebook version
├── src/
│   ├── preprocessing.py          ← cleaning, stemming, extra features (shared by all)
│   ├── train_model.py            ← split, TF-IDF, train 4 models, evaluate, save
│   ├── predict.py                ← load model, predict one ad, build explanation
│   └── sample_jobs.py            ← 5 hand-written demo ads
├── docs/
│   ├── PROJECT_REPORT.md         ← academic report sections
│   └── VIVA_QA.md                ← viva questions and answers
├── app.py                        ← Streamlit web app
├── requirements.txt
└── README.md
```

## 5. Installation on Windows (exact steps)

**Step 1 – Install Python.** Download Python 3.10, 3.11 or 3.12 from <https://www.python.org/downloads/>. In the installer, **tick "Add python.exe to PATH"**, then click *Install Now*. Check it worked by opening *Command Prompt* and typing:
```
python --version
```

**Step 2 – Create the project folder.** Create `C:\Projects\fake_job_detector` and extract/copy all project files into it.

**Step 3 – Open the folder in VS Code.** Install VS Code (<https://code.visualstudio.com>), then *File → Open Folder…* and choose `fake_job_detector`. Open the terminal with **Ctrl + `** (backtick).

**Step 4 – Create a virtual environment** (a private box for this project's libraries):
```
python -m venv venv
```

**Step 5 – Activate it:**
```
venv\Scripts\activate
```
You should now see `(venv)` at the start of the line. *If PowerShell says "running scripts is disabled"*, run this once and activate again:
```
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
venv\Scripts\activate
```
(Or open a *Command Prompt* terminal in VS Code instead of PowerShell.)

**Step 6 – Install dependencies:**
```
pip install -r requirements.txt
```

**Step 7 – Download the dataset.** Search Kaggle for **"Real / Fake Job Posting Prediction"** (you need a free Kaggle account), download `fake_job_postings.csv` and put it here:
```
fake_job_detector\data\fake_job_postings.csv
```

**Step 8 – Train the model** (first run also downloads NLTK stop words, so be online; preprocessing the full dataset can take a few minutes):
```
python -m src.train_model
```

**Step 9 – Run the web app:**
```
streamlit run app.py
```
A browser tab opens at `http://localhost:8501`. Stop the app with **Ctrl + C** in the terminal.

*(Optional notebook: `pip install notebook`, then open `notebooks/model_training.ipynb` in VS Code.)*

## 6. NLP preprocessing – every step in simple language

Implemented in `src/preprocessing.py` (`clean_text`). Example input: *"URGENT!!! Earn $500/day working from HOME. Email: a@gmail.com"*

| # | Step | What it does | Why |
|---|---|---|---|
| 1 | Handle missing text | `NaN` → `""` | An empty field must not crash the program. |
| 2 | Lowercase | `HOME` → `home` | "Earn", "EARN", "earn" are the same word. |
| 3 | Remove HTML | drops `<br>`, `<p>` … | Tags are noise. |
| 4 | Mask contacts | e-mails, links, phone numbers → `emailtoken`, `urltoken`, `phonetoken` | The *presence* of a contact detail is a clue; the exact address is not. |
| 5 | Remove punctuation & special characters + **tokenize** | keeps only runs of letters, splits into word tokens | `$500/day!!!` becomes `day`. Tokenization = cutting text into words. |
| 6 | Remove stop words | drops `the, is, and, to…` | They appear everywhere and carry little meaning. (We keep *no/not* because "**no** experience needed" matters.) |
| 7 | Stemming (Porter) | `earning, earns, earned` → `earn` | Groups word forms so the model sees one idea. (Stemming is fast and needs no extra download; lemmatization is a possible upgrade.) |
| 8 | Combine fields | title + department + company profile + description + requirements + benefits + employment type + experience + education + industry + function → **one string** | One text per ad is what TF-IDF needs. |

Result for the example: `urgent earn day work home email emailtoken` (roughly).

## 7. Feature extraction

### TF-IDF in simple words
* **TF (term frequency)** – how often a word appears in *this* ad.
* **IDF (inverse document frequency)** – how *rare* the word is across *all* ads. Words in almost every ad (`work`, `team`) get a low IDF; rare words (`wire`, `fee`) get a high IDF.
* **TF × IDF** – a word scores high when it is frequent in this ad but rare overall, i.e. when it is *distinctive*.

### Why it suits fake-job detection
Scam ads reuse distinctive vocabulary and phrases ("registration fee", "no experience", "whatsapp", "earn weekly"). TF-IDF highlights those, ignores filler words, is fast, and works well with simple, explainable models.

### How the vectorizer works
`TfidfVectorizer` (in `train_model.py`):
1. **`fit`** on training text: reads all ads, splits into words (and two-word phrases, `ngram_range=(1, 2)`), counts document frequencies.
2. **Vocabulary** = the 10,000 most frequent terms (`max_features`), ignoring terms in fewer than 5 ads (`min_df=5`, too rare) or in over 90% of ads (`max_df=0.9`, too common). Each term gets a column number.
3. **`transform`** turns every ad into a row of 10,000 numbers: the TF-IDF score of each vocabulary term (0 if absent). `sublinear_tf=True` dampens very repeated words.
4. Words never seen in training are ignored at prediction time, which is why the *fitted* vectorizer is saved (`tfidf_vectorizer.pkl`).

### Additional features already implemented (17, all scaled 0–1)
`telecommuting`, `has_company_logo`, `has_questions`, `missing_company_profile`, `missing_requirements`, `missing_benefits`, `missing_location`, `missing_salary`, `short_description`, `description_length`, `url_count`, `email_count`, `exclamation_count`, `caps_ratio`, `urgent_count`, `suspicious_phrase_count`, `salary_size`.

| Idea | Could it help? | Where in the project |
|---|---|---|
| Excessive salary claims | Yes – unrealistic pay is a classic lure | `salary_size` feature + "Unrealistic salary claim" indicator (e.g. "$5000 per week") |
| Missing company information | Yes – strongly linked to fake ads | `missing_company_profile` |
| Suspicious URLs | Yes | `url_count` (deeper analysis = future work) |
| Number of questions | Weak/mixed | `has_questions` |
| Remote-work indicator | Mixed – many real jobs are remote too | `telecommuting` |
| Extremely short descriptions | Yes | `short_description`, `description_length` |
| Urgent language | Yes | `urgent_count`, `exclamation_count`, `caps_ratio` |
| Contact patterns / e-mail domains | Yes | `email_count`, indicator for free e-mail domains, WhatsApp/Telegram, phone numbers |
| Missing location | Yes | `missing_location` |
| Company profile availability | Yes | `missing_company_profile` |

> Note: EMSCAD hides real e-mail addresses/links/phones behind placeholder tokens, so the *domain* of an e-mail cannot be learned from that dataset. The free-e-mail check is therefore a **rule-based indicator** in the app, not something the model learned.

## 8. Machine-learning models and how one is selected

All four use the same features. `class_weight="balanced"` (where supported) tells the model that mistakes on the rare FAKE class cost more.

| Model | Idea | Strengths | Weaknesses |
|---|---|---|---|
| **Logistic Regression** | Weighs each feature and adds them up | Fast, explainable, strong with TF-IDF | Only straight-line boundaries |
| **Naive Bayes** | Word probabilities per class | Very fast, good for text baselines | Assumes words are independent; probabilities can be over-confident |
| **Support Vector Machine** (LinearSVC, calibrated) | Finds the widest margin between classes | Excellent on high-dimensional text | Less explainable; needs calibration for probabilities |
| **Random Forest** | Many decision trees vote | Captures feature interactions | Slower, larger file, often weaker on sparse text |

**Selection rule** (`choose_model` in `train_model.py`): pick the highest **F1-score on the FAKE class**; if several models are within 0.01 of the best, prefer the simpler, more explainable one. The script prints the reasoning.

**Why not simply the highest accuracy?** If only about 5 of every 100 ads are fake, a useless model that answers "REAL" every time is ~95% "accurate" while catching **zero** scams. So we look at precision, recall and F1 of the fake class, and at the confusion matrix.

**Trade-off to discuss:** *high recall* catches more scams but flags more genuine jobs by mistake (annoying, could scare people away from real jobs); *high precision* raises fewer false alarms but lets more scams through. For a safety tool, recall on the fake class matters a lot, but precision must stay reasonable – F1 balances the two. The 0.5 decision threshold is one setting you can experiment with.

## 9. Evaluation metrics explained simply

Looking at the FAKE class as the "positive" class:

| Metric | Meaning | Plain-language question |
|---|---|---|
| **Accuracy** | correct predictions ÷ all predictions | "How often is the model right overall?" (misleading with imbalance) |
| **Precision** | true fakes ÷ everything flagged fake | "When it says FAKE, how often is it right?" |
| **Recall** | true fakes caught ÷ all real fakes | "Of all fake ads, how many did it catch?" |
| **F1-score** | balance of precision and recall | "One number that punishes being bad at either." |
| **Confusion matrix** | 2×2 table: real/fake × predicted real/fake | Shows the four outcomes: true real, false alarm (real flagged fake), missed fake, caught fake. |
| **ROC-AUC** | ranking quality across all thresholds | "Does it give fake ads higher scores than real ones?" |

Training writes `reports/model_comparison.csv`, `model_comparison.png`, `confusion_matrices.png`, `class_distribution.png` and `classification_reports.txt`.

## 10. Running the project

```
venv\Scripts\activate
python -m src.train_model      # trains, evaluates, saves models/ and reports/
python -m src.predict          # tests the 5 demo ads in the terminal
streamlit run app.py           # opens the web app
```

## 11. Testing with the demo ads

`src/sample_jobs.py` has **5 hand-written demo/test ads** (not verified real advertisements): 2 likely genuine, 3 suspicious.

* In the app: open *"Try a demo advertisement"*, click **Demo 1 … Demo 5**, then **Detect Fake Job**.
* In the terminal: `python -m src.predict`.
* Your own test: paste any ad into the large text box (missing fields are treated as "information missing") and click **Detect Fake Job**.

The results depend on your trained model, so this README does not promise specific percentages. A sensible outcome is that Demos 3–5 score noticeably higher fake-likelihood than Demos 1–2 and show several indicators. If not, treat that as something to investigate and discuss in your report, not as a bug to hide.

## 12. Expected output (format)

**Terminal, `python -m src.train_model`:** dataset size and class counts → duplicates removed → train/test sizes → TF-IDF vocabulary size → a comparison table with Accuracy, Precision, Recall, F1, ROC-AUC for the four models → the selected model and reasoning → a classification report.

**App, after "Detect Fake Job":**
```
Prediction:   🔴 POTENTIALLY FAKE JOB POST      (or 🟢 REAL JOB POST)
Prediction Confidence: <xx>%      Fake-likelihood score: <xx>%
Why was this prediction made?
  ⚠ Suspicious wording – phrases such as "no experience needed", …
  ⚠ Missing company information – no company profile was provided
  ⚠ Unrealistic salary claim – …
  ✅ Reassuring signs (if any)
Words linked to FAKE / REAL posts in the training data (tables)
```
Displayed probabilities are capped as ">99%" / "<1%" – a model is never 100% certain.

## 13. Limitations (be honest in your report)

* Learned from one historical, English-language dataset; new scam styles may be missed.
* Class imbalance and label noise affect results.
* Stemming is crude; the model reads word patterns, not meaning.
* Pasting a full ad without filling the form makes some fields "missing" and can push the score up.
* It cannot verify a company, website, or e-mail domain.

## 14. Basic version vs future enhancements

**Implemented (basic version):** cleaning, tokenization, stop-word removal, stemming, TF-IDF, 17 extra features, four-model comparison, saved model, Streamlit app, rule-based indicators, influential words.

**Future enhancements (NOT implemented):** DistilBERT/BERT classifier, SHAP/LIME explainable AI, URL analysis, e-mail/domain verification, company website verification, Named Entity Recognition, larger scam-keyword lists, real-time job-board monitoring, browser extension, multilingual detection, LLM-generated explanations, a database of reported scams.

## 15. More documents

* `docs/PROJECT_REPORT.md` – the 23 sections of the academic report.
* `docs/VIVA_QA.md` – likely viva questions with simple answers.
