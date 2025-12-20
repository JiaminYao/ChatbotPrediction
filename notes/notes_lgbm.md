# LightGBM Classifier (Classical ML Path)

This model uses TF-IDF feature extraction followed by a LightGBM gradient-boosted decision tree classifier to predict which response (A or B) is preferred — or whether they are equally good (tie).
It serves as a strong baseline for text-based pairwise evaluation.

---

## Model Architecture (Flow)
           
Text Fields ─► Separate TF-IDF Vectorizers (1–2 grams, 20k features each)
           ├─► Prompt → TF-IDF(prompt)
           ├─► Response A → TF-IDF(resp_a)
           ├─► Response B → TF-IDF(resp_b)
           └─► hstack([Prompt, A, B]) ─► Combined Sparse Matrix
                       ↓
           Train / Val / Test = 64% / 16% / 20%
           (derived from 80/20 split within train.csv)
                       ↓
           LightGBM Classifier (multiclass, 3 classes)


## Text Preprocessing

| **Step** | **Description** |
|-----------|-----------------|
| **Load Data** | Training and test sets are loaded and validated using `load_and_prepare_data()`. |
| **Label Encoding** | Converts one-hot labels to integers: `0` = A wins, `1` = B wins, `2` = Tie. |
| **TF-IDF Vectorization** | Uses `TfidfVectorizer` with `ngram_range=(1,2)` and up to **20,000 features** per field. |
| **Field Separation** | Vectorizes `prompt`, `response_a`, and `response_b` separately to capture unique semantics. |
| **Feature Concatenation** | Merges all fields into one sparse feature matrix using `scipy.sparse.hstack()`. |
| **Train / Validation Split** | Splits data into **80% training** and **20% validation** with `train_test_split(stratify=y)`. |

## Training Configuration

| **Parameter** | **Value / Description** |
|----------------|--------------------------|
| **Model Type** | `LightGBMClassifier` — Gradient Boosted Decision Trees |
| **Objective** | `multiclass` — 3-class classification (`A wins`, `B wins`, `Tie`) |
| **num_class** | `3` |
| **n_estimators** | `800` — total boosting rounds |
| **learning_rate** | `0.05` — step size shrinkage |
| **max_depth** | `-1` — unrestricted tree depth |
| **num_leaves** | `63` — controls model complexity |
| **min_child_samples** | `20` — minimum samples per leaf |
| **subsample** | `0.8` — row sampling for bagging |
| **colsample_bytree** | `0.8` — feature sampling per tree |
| **reg_alpha / reg_lambda** | `0.1` each — L1 / L2 regularization |
| **boosting_type** | `gbdt` — Gradient Boosting Decision Trees |
| **random_state** | `42` — ensures reproducibility |
| **n_jobs** | `-1` — uses all CPU cores |
| **force_col_wise** | `True` — optimizes memory layout for large feature matrices |
| **Eval Metric** | `multi_logloss` — multiclass log-loss objective |
| **Early Stopping** | Stops after **50 rounds** without improvement |
| **Validation Split** | `20%` of training data used for validation |
