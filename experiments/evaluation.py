from typing import Dict, List, Tuple
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, roc_auc_score, log_loss, roc_curve, auc
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import label_binarize

def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute and print overall (global) evaluation metrics.

    These are dataset-level (global) scores:
    - Accuracy: overall correctness across all samples
    - Precision (Macro): mean precision across all classes (equal weight per class)
    - Recall (Macro): mean recall across all classes
    - F1 (Macro): harmonic mean of macro precision and recall

    Returns:
        dict: containing accuracy, precision_macro, recall_macro, and f1_macro
    """
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    print("*** GLOBAL METRICS ***")
    print(f"Accuracy (Global)      : {acc:.4f}")
    print(f"Precision (Macro Avg)  : {prec:.4f}")
    print(f"Recall (Macro Avg)     : {rec:.4f}")
    print(f"F1-Score (Macro Avg)   : {f1:.4f}")

    return dict(
        accuracy=acc,
        precision_macro=prec,
        recall_macro=rec,
        f1_macro=f1,
    )


def eval_classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, class_names_map: Dict[int, str]
) -> None:
    """
    Print per-class evaluation metrics (Precision, Recall, F1, Support),
    plus macro and weighted averages at the bottom.
    """
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=[class_names_map[i] for i in sorted(class_names_map)],
        output_dict=True,
        zero_division=0,
    )

    print("\n*** PER-CLASS EVALUATION ***")
    print(f"{'Class':<20}{'Precision':>10}{'Recall':>10}{'F1-Score':>10}{'Support':>10}")
    print("-" * 60)

    for cls_name in [class_names_map[i] for i in sorted(class_names_map)]:
        vals = report_dict[cls_name]
        print(f"{cls_name:<20}{vals['precision']:>10.2f}{vals['recall']:>10.2f}{vals['f1-score']:>10.2f}{int(vals['support']):>10}")

    # Macro and Weighted averages (no redundant accuracy row)
    print("-" * 60)
    macro = report_dict["macro avg"]
    weighted = report_dict["weighted avg"]
    total_support = int(sum(v["support"] for k, v in report_dict.items() if isinstance(v, dict)))
    print(f"{'Macro Avg':<20}{macro['precision']:>10.2f}{macro['recall']:>10.2f}{macro['f1-score']:>10.2f}{total_support:>10}")
    print(f"{'Weighted Avg':<20}{weighted['precision']:>10.2f}{weighted['recall']:>10.2f}{weighted['f1-score']:>10.2f}{total_support:>10}")
    

def eval_roc_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Return macro OvR ROC-AUC (handles multiclass)."""
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y_true)
    # For safety if only one column returned (edge case)
    if y_bin.ndim == 1:
        y_bin = np.vstack([1 - y_bin, y_bin]).T
    try:
        auc_macro_ovr = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")
    except Exception:
        auc_macro_ovr = np.nan
    print("\n*** ROC-AUC EVALUATION ***")
    print(f"ROC-AUC (OvR) : {auc_macro_ovr:.4f}")
    return auc_macro_ovr

def eval_log_loss(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Return log-loss (requires correct class indices)."""
    labels = list(range(y_proba.shape[1]))
    try:
        ll = log_loss(y_true, y_proba, labels=labels)
    except Exception:
        ll = np.nan
    print("\n*** LOG-LOSS EVALUATION ***")
    print(f"Log-loss      : {ll:.4f}")
    return ll

def eval_log_loss_per_class(y_true, y_proba):
    """Compute and print per-class log-loss."""
    classes = np.unique(y_true)
    results = {}

    print("\n*** LOG-LOSS PER CLASS ***")
    for c in classes:
        mask = (y_true == c)
        if mask.sum() > 0:
            ll = log_loss(y_true[mask], y_proba[mask], labels=classes)
            results[c] = ll
            print(f"Class {c}: {ll:.4f}  (n={mask.sum()})")
        else:
            results[c] = np.nan
            print(f"Class {c}: N/A (no samples)")

    return results
    
def eval_confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    """Return numeric confusion matrix (rows=true, cols=pred)."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    print("\nConfusion Matrix (rows=true, cols=pred):\n", cm)
    return cm

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Dict[int, str],
    title: str = "Confusion Matrix",
    save_path: str = None
) -> None:
    """
    Plot a confusion matrix heatmap using numeric class labels (0, 1, 2).
    The dictionary 'class_names' maps label indices to class names, but
    only the numeric indices are shown on the plot for clarity.
    """
    plt.figure(figsize=(6, 5))
    idx = list(class_names.keys())  # e.g., [0, 1, 2]

    # Create the heatmap and store the Axes object
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=idx,  # show numeric labels
        yticklabels=idx,  # show numeric labels
    )

    # Rotate Y-axis labels vertically for better alignment
    ax.set_yticklabels(ax.get_yticklabels(), rotation=90, va="center")

    ax.set_xlabel("Predicted (class index)", fontsize=11)
    ax.set_ylabel("Actual (class index)", fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")
    else:
        plt.show()

    plt.close()

def plot_roc_curves(y_true: np.ndarray, y_proba: np.ndarray, class_names_map: Dict[int, str], title_prefix: str = "ROC", save_path=None) -> None:
    """
    Plot ROC curves:
      - per-class OvR
      - micro-average
      - macro-average
    Works for multiclass; for binary shows a single curve.
    """
    n_classes = y_proba.shape[1]
    plt.figure(figsize=(9, 7))

    if n_classes > 2:
        lb = LabelBinarizer()
        y_bin = lb.fit_transform(y_true)
        if y_bin.shape[1] == 1:  # safety
            y_bin = np.hstack([1 - y_bin, y_bin])

        fpr, tpr, roc_auc_vals = {}, {}, {}
        # Per-class
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc_vals[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], linewidth=1.75,
                     label=f"{class_names_map[i]} (AUC = {roc_auc_vals[i]:.3f})")

        # Micro-average
        fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_proba.ravel())
        roc_auc_vals["micro"] = auc(fpr["micro"], tpr["micro"])
        plt.plot(fpr["micro"], tpr["micro"], linestyle="--", linewidth=2.0,
                 label=f"micro-average (AUC = {roc_auc_vals['micro']:.3f})")

        # Macro-average
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        roc_auc_vals["macro"] = auc(all_fpr, mean_tpr)
        plt.plot(all_fpr, mean_tpr, linestyle=":", linewidth=2.0,
                 label=f"macro-average (AUC = {roc_auc_vals['macro']:.3f})")

        plt.title(f"{title_prefix} (Multiclass OvR)")
    else:
        # Binary: single curve
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        roc_auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2.0, label=f"ROC (AUC = {roc_auc_val:.3f})")
        plt.title(f"{title_prefix} (Binary)")

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.25, color="gray")
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", frameon=True); plt.grid(True, alpha=0.3)
    plt.tight_layout();
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")
    else:
        plt.show()

    plt.close()


def save_roc_to_csv(y_true, y_proba, method_name, fold_idx=None, save_dir="results/roc_perclass"):
    """
    Compute and save ROC curve data.
    Works for binary and multiclass (One-vs-Rest) settings.

    Saves one CSV per class:
        e.g. LogisticRegression_fold1_class0.csv
    """
    os.makedirs(save_dir, exist_ok=True)
    y_true = np.array(y_true)
    y_proba = np.array(y_proba)
    n_classes = y_proba.shape[1] if y_proba.ndim > 1 else 2

    # Binarize true labels
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        auc_i = auc(fpr, tpr)
        df = pd.DataFrame({"fpr": fpr, "tpr": tpr})

        if fold_idx is not None:
            filename = f"{method_name}_fold{fold_idx}_class{i}.csv"
        else:
            filename = f"{method_name}_class{i}.csv"

        path = os.path.join(save_dir, filename)
        df.to_csv(path, index=False)
        print(f"Saved ROC data for class {i} (AUC={auc_i:.4f}) → {path}")



def build_submission(test_df, y_pred_test, y_proba_test, model_name="model", out_dir="results/submission"):
    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Use 'id' if present; fallback to row index
    test_ids = test_df["id"] if "id" in test_df.columns else pd.RangeIndex(len(test_df))

    # Build submission DataFrame
    submission = pd.DataFrame({"id": test_ids, "prediction": y_pred_test})
    for i in range(y_proba_test.shape[1]):
        submission[f"proba_class_{i}"] = y_proba_test[:, i]

    # Save to CSV
    out_path = os.path.join(out_dir, f"submission_{model_name}.csv")
    submission.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    return submission