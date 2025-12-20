# data_utils.py
import ast
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------- Shared utilities ----------
def parse_prompt_cell(x):
    """Convert serialized lists like '["q1","q2"]' to plain strings."""
    if isinstance(x, list):
        return " ".join(map(str, x))
    if isinstance(x, str) and x.startswith("[") and x.endswith("]"):
        try:
            lst = ast.literal_eval(x)
            if isinstance(lst, (list, tuple)):
                return " ".join(map(str, lst))
        except Exception:
            pass
    return str(x) if x is not None else ""

def map_winner_to_label(df):
    """Map one-hot columns → integer labels {A:0, B:1, Tie:2}."""
    idx = df[["winner_model_a", "winner_model_b", "winner_tie"]].idxmax(axis=1)
    mapping = {"winner_model_a": 0, "winner_model_b": 1, "winner_tie": 2}
    return idx.map(mapping).astype(int)


def load_and_prepare_data(train_path="data/train.csv", test_path="data/test.csv"):
    """Load CSVs and ensure required columns exist."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    expected_cols = [
        "prompt", "response_a", "response_b",
        "winner_model_a", "winner_model_b", "winner_tie"
    ]
    missing = [c for c in expected_cols if c not in train_df.columns]
    if missing:
        raise ValueError(f"Missing columns in train.csv: {missing}")

    y = map_winner_to_label(train_df)
    class_names = {0: "winner_model_a", 1: "winner_model_b", 2: "winner_tie"}

    return train_df, test_df, y, class_names


# ---------- TF-IDF (classical ML) pipeline ----------
def prepare_tfidf_pipeline(train_df, test_df, y, test_size=0.2, random_state=42):
    """Prepare TF-IDF features and splits for classical ML models."""
    vec_p = TfidfVectorizer(max_features=20_000, ngram_range=(1,2), min_df=2)
    vec_a = TfidfVectorizer(max_features=20_000, ngram_range=(1,2), min_df=2)
    vec_b = TfidfVectorizer(max_features=20_000, ngram_range=(1,2), min_df=2)

    X_all = hstack([
        vec_p.fit_transform(train_df["prompt"].apply(parse_prompt_cell)),
        vec_a.fit_transform(train_df["response_a"].fillna("")),
        vec_b.fit_transform(train_df["response_b"].fillna("")),
    ])
    X_test = hstack([
        vec_p.transform(test_df["prompt"].apply(parse_prompt_cell)),
        vec_a.transform(test_df["response_a"].fillna("")),
        vec_b.transform(test_df["response_b"].fillna("")),
    ])

    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y, test_size=test_size, stratify=y, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, (vec_p, vec_a, vec_b)


# ---------- BERT/Text-based pipeline ----------
def prepare_text_pipeline(train_df, test_df, y, test_size=0.2, random_state=42):
    """
    Prepare text inputs for transformer-based models (e.g., BERT, RoBERTa).
    Handles list-like JSON cells, bytes, and ensures clean '[SEP]' joined text.

    Returns:
        X_train_text, X_val_text, X_test_text, y_train_text, y_val_text
    """

    def _maybe_list(x):
        """Parse JSON-like or ensure list wrapping."""
        if isinstance(x, list):
            return x
        if isinstance(x, str) and x.strip().startswith("[") and x.strip().endswith("]"):
            try:
                v = json.loads(x)
                if isinstance(v, list):
                    return v
            except Exception:
                pass
        return [x]

    def _to_str(x):
        """Ensure safe UTF-8 conversion."""
        if x is None:
            return ""
        if isinstance(x, bytes):
            return x.decode("utf-8", errors="ignore")
        s = str(x)
        return s.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")

    def _build_combined_row(row):
        """Construct robust 'prompt [SEP] resp_a [SEP] resp_b' strings."""
        p_list = _maybe_list(row["prompt"])
        a_list = _maybe_list(row["response_a"])
        b_list = _maybe_list(row["response_b"])
        p = " ".join(_to_str(t) for t in p_list)
        a = " ".join(_to_str(t) for t in a_list)
        b = " ".join(_to_str(t) for t in b_list)
        return f"{p} [SEP] {a} [SEP] {b}"

    # Build combined_text if not already present
    if "combined_text" not in train_df.columns:
        train_df["combined_text"] = train_df.apply(_build_combined_row, axis=1)
    if "combined_text" not in test_df.columns:
        test_df["combined_text"] = test_df.apply(_build_combined_row, axis=1)

    train_texts_all = train_df["combined_text"].astype(str).tolist()
    test_texts_all  = test_df["combined_text"].astype(str).tolist()

    # Train/val split with alignment
    X_train_text, X_val_text, y_train_text, y_val_text = train_test_split(
        train_texts_all, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Convert labels to numpy for torch/tensorflow use
    y_train_text = np.asarray(y_train_text, dtype=int)
    y_val_text   = np.asarray(y_val_text, dtype=int)

    return X_train_text, X_val_text, test_texts_all, y_train_text, y_val_text

# ---------- GEMMA/Text-based pipeline ----------
def prepare_text_pipeline_gemma(train_df, test_df, y, test_size=0.2, random_state=42):
    """
    Prepare text inputs for Gemma-style causal language models.
    Builds natural instruction-like text instead of BERT-style [SEP] joins.

    Returns:
        X_train_text, X_val_text, X_test_text, y_train_text, y_val_text
    """
    import json
    import numpy as np
    from sklearn.model_selection import train_test_split

    def _maybe_list(x):
        """Parse JSON-like or ensure list wrapping."""
        if isinstance(x, list):
            return x
        if isinstance(x, str) and x.strip().startswith("[") and x.strip().endswith("]"):
            try:
                v = json.loads(x)
                if isinstance(v, list):
                    return v
            except Exception:
                pass
        return [x]

    def _to_str(x):
        """Ensure UTF-8 safe conversion."""
        if x is None:
            return ""
        if isinstance(x, bytes):
            return x.decode("utf-8", errors="ignore")
        s = str(x)
        return s.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")

    def _build_instruction_row(row):
        """Construct text prompt for Gemma in instruction-following style."""
        p_list = _maybe_list(row["prompt"])
        a_list = _maybe_list(row["response_a"])
        b_list = _maybe_list(row["response_b"])

        p = " ".join(_to_str(t) for t in p_list)
        a = " ".join(_to_str(t) for t in a_list)
        b = " ".join(_to_str(t) for t in b_list)

        # Gemma prefers clear textual markers, not [SEP]
        text = (
            f"### Prompt:\n{p}\n\n"
            f"### Response A:\n{a}\n\n"
            f"### Response B:\n{b}\n"
        )
        return text

    # Ensure no NaNs
    train_df = train_df.fillna("")
    test_df = test_df.fillna("")

    # Build combined instruction text
    if "combined_text" not in train_df.columns:
        train_df["combined_text"] = train_df.apply(_build_instruction_row, axis=1)
    if "combined_text" not in test_df.columns:
        test_df["combined_text"] = test_df.apply(_build_instruction_row, axis=1)

    train_texts_all = train_df["combined_text"].astype(str).tolist()
    test_texts_all  = test_df["combined_text"].astype(str).tolist()

    # Stratified split if possible (skip if too few samples per class)
    try:
        X_train_text, X_val_text, y_train_text, y_val_text = train_test_split(
            train_texts_all, y, test_size=test_size, stratify=y, random_state=random_state
        )
    except ValueError:
        X_train_text, X_val_text, y_train_text, y_val_text = train_test_split(
            train_texts_all, y, test_size=test_size, random_state=random_state
        )

    # Convert labels to NumPy arrays
    y_train_text = np.asarray(y_train_text, dtype=int)
    y_val_text   = np.asarray(y_val_text, dtype=int)

    return X_train_text, X_val_text, test_texts_all, y_train_text, y_val_text


# ---------- Dual conversation (CNN) pipeline ----------
def prepare_dual_conversation_pipeline(train_df, test_df, y, test_size=0.2, random_state=42):
    """
    Prepare (conv_a, conv_b) text pairs for pairwise CNN or dual-encoder models.
    Handles list-like prompt/response fields and aligns conversation turns.
    """
    import ast
    from sklearn.model_selection import train_test_split

    def _maybe_list(cell):
        """Convert cell to list if JSON/list-like; otherwise wrap as list."""
        if isinstance(cell, list):
            return cell
        if isinstance(cell, str) and cell.strip().startswith("[") and cell.strip().endswith("]"):
            try:
                parsed = ast.literal_eval(cell)
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)
            except Exception:
                pass
        return ["" if cell is None else str(cell)]

    def _to_str(x):
        """Ensure UTF-8 string."""
        if x is None:
            return ""
        if isinstance(x, bytes):
            return x.decode("utf-8", errors="ignore")
        s = str(x)
        return s.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")

    def _build_pairs(df):
        """Construct conversation pairs (conv_a, conv_b)."""
        pairs = []
        for i in range(len(df)):
            prompts = _maybe_list(df.iloc[i]["prompt"])
            resp_a  = _maybe_list(df.iloc[i]["response_a"])
            resp_b  = _maybe_list(df.iloc[i]["response_b"])

            # pad to same length
            L = max(len(prompts), len(resp_a), len(resp_b))
            prompts += [""] * (L - len(prompts))
            resp_a  += [""] * (L - len(resp_a))
            resp_b  += [""] * (L - len(resp_b))

            ca, cb = [], []
            for j in range(L):
                ca.append(_to_str(prompts[j])); ca.append(_to_str(resp_a[j]))
                cb.append(_to_str(prompts[j])); cb.append(_to_str(resp_b[j]))
            pairs.append(("\n".join(ca), "\n".join(cb)))
        return pairs

    # Build conversations for train/test
    train_pairs = _build_pairs(train_df)
    test_pairs  = _build_pairs(test_df)

    # Split train/validation
    pairs_train, pairs_val, y_train, y_val = train_test_split(
        train_pairs, y, test_size=test_size, stratify=y, random_state=random_state
    )

    return pairs_train, pairs_val, test_pairs, y_train, y_val