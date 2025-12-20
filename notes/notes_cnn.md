# CNN Classifier (Dual-Input)

This model compares two text responses (Response A and Response B) given the same prompt, and predicts which response is better — or whether they are equally good (tie).
It’s implemented in TensorFlow /Keras using a shared-weights convolutional neural network (CNN).

---
## File Architecture 
Load & Prepare Data train.csv, test.csv -> Prepare Dual Conversation Pairs (A,B) dialogues -> Build TF Datasets make_dual_dataset() -> Initialize Dual-CNN Model get_dual_cnn_model()
-> Train Model model.fit(train_ds, val_ds) -> Callbacks EarlyStopping + ModelCheckpoint -> Save Best Model models_cnn_dual/cnn_dual_seed_X.keras -> Evaluate on Validation Set model.evaluate(val_ds) -> Test

## Model Architecture (Flow)

Input A ──► Vectorizer ─► Embedding ─► Conv1D ─► Pooling ─► GlobalMaxPool ─┐
                                                                           │
                                                                           ├─► Concatenate ─► Dropout ─► Dense(128, swish) ─► Dense(3, softmax)
                                                                           │
Input B ──► Vectorizer ─► Embedding ─► Conv1D ─► Pooling ─► GlobalMaxPool ─┘

| Layer | Input Shape | Output Shape | Role |
|:------|:-------------:|:--------------:|:------|
| **Conv1D(32, 3)** | (512, 64) | (510, 32) | Extract 3-gram local semantics (“is good”, “not bad”) |
| **Conv1D(32, 3)** | (510, 32) | (508, 32) | Capture deeper local relationships |
| **MaxPooling1D** | (508, 32) | (254, 32) | Downsample and retain main features |
| **Conv1D(64, 3)** | (254, 32) | (252, 64) | Extract higher-level abstract patterns |
| **GlobalMaxPooling1D** | (252, 64) | (64,) | Aggregate into sentence-level feature vector |

This Dual-CNN model features a symmetric two-branch architecture:
* The left and right branches share a vocabulary and convolutional weights to fairly extract semantic features for both A and B classifications;
* After merging, the quality of the two branches is compared using a fully connected layer;
* The final output is a three-class classification result.
  
## Text Preprocessing

| **Step** | **Description** |
|-----------|-----------------|
| **Prompt & Response Pairing** | Prompts and responses are combined into paired texts `(conv_a, conv_b)`. |
| **Shared Vectorizer** | A single `TextVectorization` layer learns a shared vocabulary of **20,000 tokens**. |
| **Tokenization & Padding** | Texts are tokenized, truncated, and padded to a maximum length of **512**. |

## Training Configuration

| **Parameter** | **Value / Description** |
|----------------|--------------------------|
| **Optimizer** | `Adam` — adaptive learning-rate optimizer combining **Momentum** and **RMSProp** for fast, stable convergence. |
| **Learning Rate** | `1e-3` |
| **Loss Function** | `sparse_categorical_crossentropy` |
| **Metrics** | `accuracy` |
| **Batch Size** | `128` |
| **Epochs** | Up to `30` (uses **EarlyStopping** for best epoch selection) |
| **Callbacks** | `EarlyStopping`, `ModelCheckpoint` |
| **Seeds** | `[42, 119, 2020, 2024, 2028]` — used for reproducibility and ensemble averaging |
| **Model Save Path** | `models_cnn_dual/cnn_dual_seed_*.keras` |
| **Validation Split** | `20%` of training data (via `train_test_split`) |
