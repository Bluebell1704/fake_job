# PROJECT REPORT

## 1. Project Title
**NLP-Based Fake Job Post Detector**

## 2. Abstract
Online job portals are increasingly misused to publish fraudulent advertisements that collect fees, personal data or bank details from job seekers. This project presents a screening system that uses Natural Language Processing (NLP) and Machine Learning to classify a job advertisement as *real* or *potentially fake*. Job-post fields are combined into one text, cleaned (lowercasing, punctuation and stop-word removal, tokenization, stemming), and converted to numbers with TF-IDF. Seventeen additional features (missing company profile, short description, urgent language, etc.) are added. Four classifiers – Logistic Regression, Naive Bayes, Support Vector Machine and Random Forest – are trained on the EMSCAD dataset and compared using accuracy, precision, recall, F1-score and confusion matrices, with special attention to class imbalance. The selected model and vectorizer are saved with Joblib and used by a Streamlit web application that shows the prediction, a confidence value and understandable warning indicators. The system is a decision-support aid and does not guarantee that an advertisement is genuine or fraudulent.

## 3. Introduction
Millions of people search for jobs online. Fraudsters exploit this by posting attractive but false offers. Manual checking of every advertisement is impossible for large portals, so automatic screening is valuable. Fake advertisements often show recognisable language patterns and missing information, which makes them suitable for text classification.

## 4. Problem Statement
Given the textual and structured content of a job advertisement, predict whether it is real or potentially fake and explain the main reasons, so that a job seeker can decide what to verify.

## 5. Existing System
Job portals mainly rely on user reports, manual moderation, basic keyword filters and account verification. Job seekers rely on their own judgement and awareness.

## 6. Limitations of Existing System
- Manual moderation is slow and does not scale.
- Fixed keyword lists are easy for scammers to avoid.
- Reports arrive only after people have already been harmed.
- Job seekers, especially freshers, often lack the experience to spot warning signs.
- Most systems give no explanation of why a post is suspicious.

## 7. Proposed System
A web application where the user enters or pastes a job advertisement. The system preprocesses the text, computes TF-IDF and structured features, predicts with a trained classifier, and shows (a) REAL / POTENTIALLY FAKE, (b) a confidence percentage, and (c) explanations: rule-based warning indicators and words most associated with fake or real posts in the training data.

## 8. Objectives
1. Build an NLP pipeline that converts job text into numerical features.
2. Train and compare four classifiers on an imbalanced dataset.
3. Select a model using fake-class F1/recall rather than accuracy alone.
4. Provide explainable output and a simple web interface.
5. Communicate clearly that the system is a screening aid, not proof.

## 9. Scope
The system handles English-language job advertisements in the format of the EMSCAD dataset. It classifies text and basic metadata only. It does not verify companies, websites or e-mail domains, and does not monitor job boards automatically.

## 10. Methodology
1. Collect dataset (EMSCAD `fake_job_postings.csv`).
2. Clean data: handle missing values, remove duplicates.
3. Combine text fields; preprocess text.
4. Stratified 80/20 train-test split (`random_state=42`).
5. Fit TF-IDF on the training set only; add structured features.
6. Train four classifiers with class-imbalance handling.
7. Evaluate on the test set; select model.
8. Save vectorizer and model (Joblib).
9. Build Streamlit application with explanations.

## 11. System Architecture
Job Posting → Data Collection → Data Cleaning → Text Preprocessing → Tokenization → Stop-word Removal → Stemming → TF-IDF Feature Extraction → Machine Learning Model → Prediction → Confidence Score → Explanation → Streamlit Interface. (See the diagram in `README.md`.)

## 12. Hardware Requirements
- Processor: Intel Core i3 or equivalent (i5 recommended)
- RAM: 4 GB minimum (8 GB recommended for training on the full dataset)
- Storage: 1 GB free
- Internet connection for installing libraries and downloading the dataset

## 13. Software Requirements
- Windows 10/11 (Linux/macOS also work)
- Python 3.10 – 3.12
- Libraries: pandas, numpy, scipy, scikit-learn, nltk, matplotlib, seaborn, streamlit, joblib
- VS Code (or any editor), a web browser

## 14. Dataset Description
EMSCAD (Employment Scam Aegean Dataset), University of the Aegean; commonly distributed on Kaggle as `fake_job_postings.csv`. Each row is a job advertisement with fields such as title, location, department, salary range, company profile, description, requirements, benefits, employment type, required experience, required education, industry, function, telecommuting, has company logo, has questions and the target `fraudulent` (1 = fake, 0 = real). The dataset is imbalanced (fake ads are a small minority). *Insert here the exact counts printed by your training run:* total posts = ___, real = ___, fake = ___, duplicates removed = ___, train = ___, test = ___.

## 15. NLP Techniques Used
- Missing-value handling and field combination
- Lowercasing
- HTML, punctuation and special-character removal
- Masking of e-mail, link and phone patterns
- Tokenization (NLTK `RegexpTokenizer`)
- Stop-word removal (NLTK list; "no/not" kept)
- Stemming (Porter Stemmer)
- TF-IDF with unigrams and bigrams
- Hand-crafted features: missing information flags, description length, URL/e-mail counts, urgency and suspicious-phrase counts, capital-letter ratio, salary size

## 16. Machine Learning Algorithms
- **Logistic Regression** – linear model that estimates probabilities; explainable.
- **Multinomial Naive Bayes** – probabilistic model based on word frequencies.
- **Support Vector Machine (LinearSVC with probability calibration)** – maximum-margin classifier, strong on high-dimensional text.
- **Random Forest** – ensemble of decision trees.

## 17. Implementation
- `src/preprocessing.py` – cleaning and feature functions shared by training and prediction.
- `src/train_model.py` – split, TF-IDF, training, evaluation, model selection, saving models and reports.
- `src/predict.py` – loads the saved files, predicts, produces indicators and influential words.
- `app.py` – Streamlit interface (input form, full-ad text box, "Detect Fake Job" button, prediction, confidence, "Why was this prediction made?").
- Data-leakage controls: duplicates removed before splitting; vectorizer fitted on training data only; ID column unused.
- Imbalance handling: stratified split, class weights, evaluation on fake-class precision/recall/F1.

## 18. Results
*Fill this section with your own results after training (from `reports/model_comparison.csv`, `confusion_matrices.png`, `classification_reports.txt`). Do not copy numbers from other sources.*

| Model | Accuracy | Precision (fake) | Recall (fake) | F1 (fake) | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | | | | | |
| Naive Bayes | | | | | |
| Support Vector Machine | | | | | |
| Random Forest | | | | | |

Discussion points: which model was selected and why; how accuracy compares with fake-class recall; number of missed fake posts and false alarms in the confusion matrix; behaviour on the five demo advertisements (clearly labelled as demo examples).

## 19. Advantages
- Fast, automatic first-level screening.
- Uses both text and structural clues.
- Explains its output in understandable language.
- Beginner-friendly, open-source tools; runs on an ordinary laptop.
- Modular code that is easy to extend.

## 20. Limitations
- Depends on the quality, age and language of the training data.
- Class imbalance and label noise limit reliability.
- May misclassify unusual genuine posts and well-written scams.
- Cannot verify companies, links or e-mail domains.
- Stemming and TF-IDF ignore deeper meaning and word order beyond two-word phrases.
- Not proof of fraud; results must be treated as decision support.

## 21. Future Enhancements
*(Not part of the implemented version.)* DistilBERT/BERT-based classification, explainable AI (SHAP/LIME), URL analysis, e-mail/domain verification, company-website verification, Named Entity Recognition, larger scam-keyword lists, real-time job-board monitoring, browser extension, multilingual detection, LLM-based explanations, a database of previously detected scams.

## 22. Conclusion
The project shows that NLP with TF-IDF features and classical machine-learning models can screen job advertisements for patterns associated with fraud, and that an explainable, cautious interface (potentially fake, confidence, indicators) makes the output more useful and responsible. The system is intended to help users decide what to verify, not to replace human judgement.

## 23. References
1. Vidros, S., Kolias, C., Kambourakis, G., Akoglu, L. (2017). *Automatic Detection of Online Recruitment Frauds: Characteristics, Methods, and a Public Dataset.* Future Internet, 9(1), 6. (EMSCAD dataset paper – please verify the citation details before submission.)
2. Employment Scam Aegean Dataset (EMSCAD), University of the Aegean, Laboratory of Information & Communication Systems Security.
3. Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12.
4. Bird, S., Klein, E., Loper, E. (2009). *Natural Language Processing with Python.* O'Reilly (NLTK).
5. Streamlit documentation – https://docs.streamlit.io
6. scikit-learn documentation: TfidfVectorizer, LogisticRegression, MultinomialNB, LinearSVC, RandomForestClassifier.
