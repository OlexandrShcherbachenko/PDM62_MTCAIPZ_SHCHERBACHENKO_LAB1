# LR_1_task_5.py —
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, accuracy_score, recall_score, precision_score, f1_score,
    roc_curve, roc_auc_score
)

# ---------- Дані ----------
df = pd.read_csv('data_metrics.csv')
y_true = df.actual_label.values.astype(int)
p_rf   = df.model_RF.values.astype(float)
p_lr   = df.model_LR.values.astype(float)

# Базовий поріг для первинного порівняння
BASE_THR = 0.5
y_rf = (p_rf >= BASE_THR).astype(int)
y_lr = (p_lr >= BASE_THR).astype(int)

# ---------- ВЛАСНІ ФУНКЦІЇ (без sklearn) ----------
def find_TP(y_true, y_pred): return int(((y_true == 1) & (y_pred == 1)).sum())
def find_FN(y_true, y_pred): return int(((y_true == 1) & (y_pred == 0)).sum())
def find_FP(y_true, y_pred): return int(((y_true == 0) & (y_pred == 1)).sum())
def find_TN(y_true, y_pred): return int(((y_true == 0) & (y_pred == 0)).sum())

def shcherbachenko_confusion_matrix(y_true, y_pred):
    TP, FN, FP, TN = find_TP(y_true, y_pred), find_FN(y_true, y_pred), find_FP(y_true, y_pred), find_TN(y_true, y_pred)
    # формат як у sklearn: [[TN, FP], [FN, TP]]
    return np.array([[TN, FP], [FN, TP]])

def shcherbachenko_accuracy_score(y_true, y_pred):
    TP, FN, FP, TN = find_TP(y_true, y_pred), find_FN(y_true, y_pred), find_FP(y_true, y_pred), find_TN(y_true, y_pred)
    denom = TP + TN + FP + FN
    return (TP + TN) / denom if denom else 0.0

def shcherbachenko_precision_score(y_true, y_pred):
    TP, FP = find_TP(y_true, y_pred), find_FP(y_true, y_pred)
    return TP / (TP + FP) if (TP + FP) else 0.0

def shcherbachenko_recall_score(y_true, y_pred):
    TP, FN = find_TP(y_true, y_pred), find_FN(y_true, y_pred)
    return TP / (TP + FN) if (TP + FN) else 0.0

def shcherbachenko_f1_score(y_true, y_pred):
    p = shcherbachenko_precision_score(y_true, y_pred)
    r = shcherbachenko_recall_score(y_true, y_pred)
    return (2 * p * r) / (p + r) if (p + r) else 0.0

# ---------- Порівняння з sklearn для базового порога ----------
def report(name, y_pred):
    print(f"\n=== {name} (threshold={BASE_THR}) ===")
    print("Моя confusion_matrix:\n", shcherbachenko_confusion_matrix(y_true, y_pred))
    print("sklearn confusion_matrix:\n", confusion_matrix(y_true, y_pred))
    print(f"Accuracy:  my={shcherbachenko_accuracy_score(y_true, y_pred):.4f}  "
          f"skl={accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: my={shcherbachenko_precision_score(y_true, y_pred):.4f}  "
          f"skl={precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    my={shcherbachenko_recall_score(y_true, y_pred):.4f}    "
          f"skl={recall_score(y_true, y_pred):.4f}")
    print(f"F1:        my={shcherbachenko_f1_score(y_true, y_pred):.4f}        "
          f"skl={f1_score(y_true, y_pred):.4f}")

report("RF", y_rf)
report("LR", y_lr)

# ---------- ROC-криві й AUC ----------
fpr_RF, tpr_RF, _ = roc_curve(y_true, p_rf)
fpr_LR, tpr_LR, _ = roc_curve(y_true, p_lr)
auc_RF = roc_auc_score(y_true, p_rf)
auc_LR = roc_auc_score(y_true, p_lr)

plt.figure()
plt.plot(fpr_RF, tpr_RF, label=f'RF AUC: {auc_RF:.3f}')
plt.plot(fpr_LR, tpr_LR, label=f'LR AUC: {auc_LR:.3f}')
plt.plot([0,1], [0,1], '--', label='random')
plt.title('ROC Curves')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()

# водяні підписи
fig = plt.gcf()
for (x, y, ha, va) in [(0.02,0.02,'left','bottom'), (0.98,0.02,'right','bottom'),
                       (0.02,0.98,'left','top'),   (0.98,0.98,'right','top')]:
    fig.text(x, y, "Shcherbachenko", ha=ha, va=va, fontsize=8, alpha=0.6)

plt.savefig("LR_1_task_5_ROC.png", dpi=150, bbox_inches="tight")
plt.show()

# ---------- Порівняння різних порогів ----------
thresholds = [0.3, 0.5, 0.7]
print("\n=== Порівняння порогів для RF і LR (Acc/Prec/Rec/F1) ===")
for thr in thresholds:
    y_rf_thr = (p_rf >= thr).astype(int)
    y_lr_thr = (p_lr >= thr).astype(int)
    print(f"\nПоріг {thr:.2f}")
    print(f"RF: Acc={shcherbachenko_accuracy_score(y_true,y_rf_thr):.3f}  "
          f"Prec={shcherbachenko_precision_score(y_true,y_rf_thr):.3f}  "
          f"Rec={shcherbachenko_recall_score(y_true,y_rf_thr):.3f}  "
          f"F1={shcherbachenko_f1_score(y_true,y_rf_thr):.3f}")
    print(f"LR: Acc={shcherbachenko_accuracy_score(y_true,y_lr_thr):.3f}  "
          f"Prec={shcherbachenko_precision_score(y_true,y_lr_thr):.3f}  "
          f"Rec={shcherbachenko_recall_score(y_true,y_lr_thr):.3f}  "
          f"F1={shcherbachenko_f1_score(y_true,y_lr_thr):.3f}")

# ---------- Автопошук найкращого порога за максимумом F1 ----------
def best_threshold_by_f1(y_true, proba, grid=None):
    if grid is None:
        grid = np.linspace(0.0, 1.0, 201)  # крок 0.005
    best_thr, best_f1 = 0.5, -1.0
    for thr in grid:
        y_pred = (proba >= thr).astype(int)
        f1v = shcherbachenko_f1_score(y_true, y_pred)
        if f1v > best_f1:
            best_f1, best_thr = f1v, thr
    return best_thr, best_f1

best_thr_rf, best_f1_rf = best_threshold_by_f1(y_true, p_rf)
best_thr_lr, best_f1_lr = best_threshold_by_f1(y_true, p_lr)

print("\n=== Найкращі пороги за F1 ===")
print(f"RF: best_thr={best_thr_rf:.3f}, best_F1={best_f1_rf:.3f}")
print(f"LR: best_thr={best_thr_lr:.3f}, best_F1={best_f1_lr:.3f}")
