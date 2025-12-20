# Logistic Regression Classifier (Classical ML Baseline)

This model uses TF-IDF features combined with a multinomial Logistic Regression classifier to predict which response (A or B) is preferred — or whether they are equally good (tie).  
It serves as a simple yet strong linear baseline for multiclass text classification.

---

## Model Architecture (Flow)

Text Fields ─► TF-IDF Vectorizer (1–2 grams, 20k features each)
         **For training**
           ├─► Prompt TF-IDF
           ├─► Response A TF-IDF
           ├─► Response B TF-IDF
           └─► hstack() Use fit+transform ─► Feature Matrix ─► Logistic Regression Classifier
           **For testing**
           ├─► Prompt TF-IDF
           ├─► Response A TF-IDF
           ├─► Response B TF-IDF
           └─► hstack() Use transform ─► Feature Matrix ─► Logistic Regression Classifier


## Text Preprocessing

| **Step** | **Description** |
|-----------|-----------------|
| **Load Data** | Training and test sets are loaded and validated using `load_and_prepare_data()`. |
| **TF-IDF Vectorization** | Each of `prompt`, `response_a`, and `response_b` is vectorized separately using `TfidfVectorizer`. |
| **Feature Concatenation** | Sparse TF-IDF matrices are combined using `scipy.sparse.hstack()`. |
| **Label Encoding** | Converts one-hot outcome columns into integer labels: `0` = A wins, `1` = B wins, `2` = Tie. |
| **Train / Validation Split** | `train_test_split` with `test_size=0.2` and `stratify=y` ensures balanced class distribution. |


## Training Configuration

| **Parameter** | **Value / Description** |
|----------------|--------------------------|
| **Model Type** | `LogisticRegression` — Multinomial logistic regression (softmax output). |
| **Solver** | `saga` — optimized for large, sparse TF-IDF matrices. |
| **multi_class** | `multinomial` — enables true softmax across three output classes. |
| **Max Iterations** | `3000` — allows sufficient convergence for high-dimensional features. |
| **Regularization Strength (C)** | `1.0` — inverse of regularization parameter (smaller = stronger regularization). |
| **Penalty** | `l2` — default regularization type (ridge). |
| **n_jobs** | `-1` — uses all CPU cores for parallel processing. |
| **Verbose** | `0` — silent mode (set >0 for more logging). |
| **Random State** | Not fixed here (set manually for reproducibility if needed). |
| **Loss Function** | Multinomial cross-entropy (via softmax). |
