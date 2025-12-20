# BERT Classifier (Transformer Fine-Tuning)

This model fine-tunes a pretrained BERT transformer (`bert-base-uncased`) to predict which response (A or B) is preferred — or whether they are equally good (tie).  
It leverages deep contextual embeddings and self-attention mechanisms to capture semantic differences between paired texts.

---

## Model Architecture (Flow)
Text Input ─► Tokenizer (BERT WordPiece)
            └─► Convert to Input IDs + Attention Masks
                        │
                        ▼
                BERT Encoder (12 Transformer Layers)
                        │
                        ▼
                [CLS] Token Representation
                        │
                        ▼
          Dense Layer (hidden_size → 3 logits)
                        │
                        ▼
              Softmax Activation (3 classes)
                        │
                        ├─► Class 0 → winner_model_a
                        ├─► Class 1 → winner_model_b
                        └─► Class 2 → winner_tie

## Text Preprocessing

| **Step** | **Description** |
|-----------|-----------------|
| **Load Data** | Training and test sets are loaded using `load_and_prepare_data()`. |
| **Text Preparation** | Prompts and responses are combined into unified text sequences using `prepare_text_pipeline()`. |
| **Label Encoding** | Labels are mapped as: `0 = winner_model_a`, `1 = winner_model_b`, `2 = winner_tie`. |
| **Train / Validation Split** | `train_test_split` with `stratify=y` ensures balanced class distribution. |
| **Tokenization** | `AutoTokenizer.from_pretrained("bert-base-uncased")` handles subword tokenization and truncation to 512 tokens. |
| **Dynamic Padding** | Uses `DataCollatorWithPadding` for efficient batch-wise padding. |

## Training Configuration

| **Parameter** | **Value / Description** |
|----------------|--------------------------|
| **Trainer Framework** | Hugging Face `Trainer` with `TrainingArguments`. |
| **Train Batch Size** | `16` |
| **Eval Batch Size** | `32` |
| **Epochs** | `3` — typical for transformer fine-tuning. |
| **Optimizer / LR Scheduler** | `AdamW` with linear decay and warmup. |
| **Learning Rate** | `2e-5` |
| **Weight Decay** | `0.01` |
| **Warmup Ratio** | `0.06` — helps stabilize early training. |
| **Evaluation Strategy** | `epoch` — evaluates once per epoch. |
| **Save Strategy** | `epoch` — saves best model checkpoint each epoch. |
| **Metric for Best Model** | `eval_loss` (lowest = best). |
| **Early Stopping** | `load_best_model_at_end=True` ensures best checkpoint is restored. |
| **Max Checkpoints** | `save_total_limit=2` |
| **Logging Steps** | `50` |
| **Mixed Precision** | `bf16=True`, `tf32=True` (auto-skipped if unsupported). |
| **Seed** | `42` — ensures reproducibility. |


| Model               | Params | Highlights                    | Strengths                        | Recommended Use                |
| ------------------- | ------- | ----------------------------- | -------------------------------- | ------------------------------ |
| **BERT-base**       | 110M    | Original baseline model       | Fast, low memory, stable         | ✅ Good baseline               |
| **RoBERTa-base**    | 125M    | Improved pretraining          | Better context understanding     | ✅ Balanced choice             |
| **RoBERTa-large**   | 355M    | Deeper and wider              | Higher accuracy, heavy compute   | ⚠️ Only if >50K samples        |
| **DeBERTa-v3-base** | 185M    | Disentangled attention        | Strong accuracy, efficient       | ⭐ Best overall base           |
| **DeBERTa-v3-large**| 435M    | SOTA contextualization        | Highest accuracy, large memory   | ⚠️ Use with 24–32GB VRAM       |
