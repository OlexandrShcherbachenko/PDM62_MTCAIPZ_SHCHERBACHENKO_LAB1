# LR_1_task_5.py — Метрики + ROC (з власною реалізацією F1)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, accuracy_score, recall_score, precision_score, f1_score,
    roc_curve, roc_auc_score
)

df = pd.read_csv('data_metrics.csv')

y_true = df.actual_label.values.astype(int)
y_rf   = (df.model_RF >= 0.5).astype(int)
y_lr   = (df.model_LR >= 0.5).astype(int)

# ========= ВЛАСНІ ФУНКЦІЇ (без sklearn) =========
def find_TP(y_true, y_pred): return int(((y_true == 1) & (y_pred == 1)).sum())
def find_FN(y_true, y_pred): return int(((y_true == 1) & (y_pred == 0)).sum())
def find_FP(y_true, y_pred): return int(((y_true == 0) & (y_pred == 1)).sum())
def find_TN(y_true, y_pred): return int(((y_true == 0) & (y_pred == 0)).sum())

def my_confusion_matrix(y_true, y_pred):
    TP, FN, FP, TN = find_TP(y_true, y_pred), find_FN(y_true, y_pred), find_FP(y_true, y_pred), find_TN(y_true, y_pred)
    return np.array([[TN, FP], [FN, TP]])

def my_accuracy(y_true, y_pred):
    TP, FN, FP, TN = find_TP(y_true, y_pred), find_FN(y_true, y_pred), find_FP(y_true, y_pred), find_TN(y_true, y_pred)
    return (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) else 0.0

def my_precision(y_true, y_pred):
    TP, FP = find_TP(y_true, y_pred), find_FP(y_true, y_pred)
    return TP / (TP + FP) if (TP + FP) else 0.0

def my_recall(y_true, y_pred):
    TP, FN = find_TP(y_true, y_pred), find_FN(y_true, y_pred)
    return TP / (TP + FN) if (TP + FN) else 0.0

def my_f1(y_true, y_pred):
    p = my_precision(y_true, y_pred)
    r = my_recall(y_true, y_pred)
    return (2 * p * r) / (p + r) if (p + r) else 0.0

# ========= ПОРІВНЯННЯ З SKLEARN (для RF і LR) =========
def report(name, y_pred):
    print(f"\n=== {name} (threshold=0.5) ===")
    print("Моя confusion_matrix:\n", my_confusion_matrix(y_true, y_pred))
    print("sklearn confusion_matrix:\n", confusion_matrix(y_true, y_pred))
    print(f"Accuracy:  my={my_accuracy(y_true, y_pred):.4f}  skl={accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: my={my_precision(y_true, y_pred):.4f}  skl={precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    my={my_recall(y_true, y_pred):.4f}    skl={recall_score(y_true, y_pred):.4f}")
    print(f"F1:        my={my_f1(y_true, y_pred):.4f}        skl={f1_score(y_true, y_pred):.4f}")

report("RF", y_rf)
report("LR", y_lr)

# ========= ROC-криві + AUC =========
fpr_RF, tpr_RF, _ = roc_curve(df.actual_label.values, df.model_RF.values)
fpr_LR, tpr_LR, _ = roc_curve(df.actual_label.values, df.model_LR.values)
auc_RF = roc_auc_score(df.actual_label.values, df.model_RF.values)
auc_LR = roc_auc_score(df.actual_label.values, df.model_LR.values)

plt.figure()
plt.plot(fpr_RF, tpr_RF, label=f'RF AUC: {auc_RF:.3f}')
plt.plot(fpr_LR, tpr_LR, label=f'LR AUC: {auc_LR:.3f}')
plt.plot([0,1], [0,1], '--', label='random')
plt.title('ROC Curves')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()

fig = plt.gcf()
for (x, y, ha, va) in [(0.02,0.02,'left','bottom'), (0.98,0.02,'right','bottom'),
                       (0.02,0.98,'left','top'),   (0.98,0.98,'right','top')]:
    fig.text(x, y, "Shcherbachenko", ha=ha, va=va, fontsize=8, alpha=0.6)

plt.savefig("LR_1_task_5_ROC.png", dpi=150, bbox_inches="tight")
plt.show()
