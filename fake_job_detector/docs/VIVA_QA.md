# Viva Questions and Answers (simple language)

**1. What does your project do?**
It reads a job advertisement, analyses the text with NLP, and predicts whether it is real or potentially fake. It shows a confidence percentage and the warning signs behind the prediction.

**2. What is NLP?**
Natural Language Processing is the area of AI that lets computers work with human language – cleaning text, splitting it into words, and turning it into numbers a model can learn from.

**3. Why do we need text preprocessing?**
Raw text is messy (capital letters, symbols, filler words). Cleaning makes different forms of the same word look the same, so the model learns real patterns instead of noise.

**4. What is tokenization?**
Cutting a text into small units, here words. "Earn money fast" → `earn`, `money`, `fast`.

**5. What are stop words and why remove them?**
Very common words like *the, is, and*. They carry little meaning. We keep "no" and "not" because "no experience needed" is an important scam pattern.

**6. Stemming vs lemmatization?**
Stemming chops word endings by rules (`earning → earn`); it is fast but crude. Lemmatization uses a dictionary to return a real base word (`better → good`); it is more accurate but slower and needs extra data. We used the Porter stemmer for simplicity.

**7. What is TF-IDF?**
A score for each word in an ad: TF = how often it appears in this ad; IDF = how rare it is across all ads. Distinctive words get high scores; common words get low scores.

**8. Why TF-IDF instead of simple word counts?**
Counts overrate common words. TF-IDF pushes up distinctive words such as "fee" or "whatsapp" and pushes down words that appear everywhere.

**9. How is the vocabulary created?**
`TfidfVectorizer.fit` reads the training ads, collects words and two-word phrases, drops very rare and very common ones, and keeps the top 10,000. Each gets a column number. New words unseen in training are ignored at prediction time.

**10. What is the target variable?**
`fraudulent` – 1 means fake, 0 means real.

**11. What is class imbalance and why does it matter?**
Fake ads are far fewer than real ones. A model that always says "real" would look highly accurate but catch no scams. So we use stratified splitting, class weights, and judge the models by precision, recall and F1 of the fake class.

**12. What is data leakage and how did you avoid it?**
Leakage is when information from the test data influences training. We removed duplicate ads before splitting, fitted TF-IDF on training data only, and did not use the ID column.

**13. Why split the data 80/20 and why `random_state=42`?**
Training data teaches the model; test data checks it on unseen ads. A fixed random state makes the split repeatable. `stratify` keeps the real:fake ratio the same in both parts.

**14. Explain precision, recall and F1.**
Precision: when the model says FAKE, how often is it right? Recall: of all truly fake ads, how many did it catch? F1: one number balancing both.

**15. What is a confusion matrix?**
A 2×2 table counting true real, false alarms (real flagged fake), missed fakes, and caught fakes.

**16. Which model did you select and why?**
The training script selects by F1 of the fake class (preferring the simpler model when scores are within 0.01). Write here the model chosen in *your* run and quote your own numbers from `reports/model_comparison.csv`. Mention the trade-off between catching more scams (recall) and raising fewer false alarms (precision).

**17. Why not choose the model with the highest accuracy?**
Because of class imbalance – accuracy can hide a model that misses most fake ads.

**18. How does the app explain a prediction?**
Two ways: (1) rule-based indicators – suspicious phrases, missing company information, unrealistic pay claims, urgency, missing requirements, suspicious contact details, unusual description patterns; (2) words in the ad that were strongly linked to fake or real posts in the training data. They are indicators, not proof.

**19. What does "confidence" mean?**
The model's estimated probability for the predicted class, based on patterns in historical data. It is not a guarantee, so the app never shows 100%.

**20. Why "Potentially Fake" and not "Fake"?**
Because a machine-learning prediction can be wrong. The tool supports human decisions; it cannot prove fraud.

**21. Why did you save the vectorizer as well as the model?**
The model only understands the numbers produced by the exact vectorizer used in training. Prediction must use the same fitted vectorizer.

**22. What are the limitations of your system?**
Old, English-only dataset; new scam styles may be missed; imbalance and label noise; cannot verify companies, websites or e-mail domains; crude stemming.

**23. How could you improve it?**
Use DistilBERT/BERT, SHAP/LIME explanations, URL and e-mail-domain checks, company verification, NER, multilingual support, and continuous monitoring with user feedback.

**24. What extra features did you add beyond TF-IDF and why?**
Missing company profile/requirements/location/salary, short description, URL and e-mail counts, urgency and suspicious-phrase counts, capital-letter ratio, salary size, logo/questions/remote flags – because scam ads often differ in structure as well as wording.

**25. How do you handle wrong user input?**
Empty or too-short text produces a friendly warning instead of a crash; missing fields become empty strings; very long input is truncated; unexpected errors are caught and shown as a message.
