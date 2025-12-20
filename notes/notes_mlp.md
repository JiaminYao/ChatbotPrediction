# MLP Classifier (Dense Neural Network on TF-IDF + SVD)

This model uses TF-IDF features reduced via Truncated SVD (Latent Semantic Analysis), followed by a Multi-Layer Perceptron (MLP) neural network trained to predict which response (A or B) is preferred — or whether they are equally good (tie).
It serves as a hybrid approach combining classical text features with neural representations.

---

## Model Architecture (Flow)
Text Fields ─► Separate TF-IDF Vectorizers (1–2 grams, 20k each)
           ├─► Prompt TF-IDF
           ├─► Response A TF-IDF
           ├─► Response B TF-IDF
           └─► hstack([Prompt, A, B]) ─► Sparse Matrix
                       ↓
             Truncated SVD (512 dims, dense embedding)
                       ↓
             StandardScaler (normalize features)
                       ↓
             Train / Val / Test = 64% / 16% / 20%
                       ↓
             Dense MLP Classifier (ReLU activations, dropout, softmax output)


## Text Preprocessing

| **Step** | **Description** |
|-----------|-----------------|
| **Load Data** | Training and test sets (`train_df`, `test_df`, `y`, `class_names`) are loaded and verified using `load_and_prepare_data()`. |
| **TF-IDF Vectorization** | Each of `prompt`, `response_a`, and `response_b` is vectorized separately using `TfidfVectorizer(ngram_range=(1,2), max_features=20_000)`. |
| **Feature Concatenation** | Sparse TF-IDF matrices are horizontally stacked using `scipy.sparse.hstack()`. |
| **Train / Validation Split** | `train_test_split` ensures **80% training** and **20% validation** with class stratification. |
| **Dimensionality Reduction** | `TruncatedSVD(n_components=512)` reduces sparse features to dense 512-dimensional vectors. |
| **Standardization** | `StandardScaler()` is applied on SVD features (fit on training data only). |

## Training Configuration

| **Parameter** | **Value / Description** |
|----------------|--------------------------|
| **Model Type** | `MLPClassifier` — Multi-layer perceptron (feedforward neural network). |
| **Hidden Layers** | `(512, 256)` — two fully connected layers. |
| **Activation Function** | `ReLU` — introduces non-linearity for richer feature learning. |
| **Solver / Optimizer** | `adam` — adaptive learning rate combining **Momentum** and **RMSProp**. |
| **Regularization (`alpha`)** | `1e-4` — L2 regularization to prevent overfitting. |
| **Batch Size** | `256` — number of samples per gradient update. |
| **Learning Rate Init** | `1e-3` — initial learning rate for Adam optimizer. |
| **Max Iterations** | `60` — maximum training epochs (stops earlier if converged). |
| **Early Stopping** | `True` — stops training when validation score stops improving. |
| **n_iter_no_change** | `5` — patience for early stopping before halting training. |
| **Random State** | `42` — ensures reproducibility. |
| **Verbose** | `False` — suppresses training logs for cleaner output. |
