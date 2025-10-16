
# utilities.py
import numpy as np
import matplotlib.pyplot as plt

STUDENT_NAME = "Shcherbachenko"

def _add_watermarks(fig):
    fig.text(0.02, 0.02, STUDENT_NAME, ha='left',  va='bottom', fontsize=8, alpha=0.6)
    fig.text(0.98, 0.02, STUDENT_NAME, ha='right', va='bottom', fontsize=8, alpha=0.6)
    fig.text(0.02, 0.98, STUDENT_NAME, ha='left',  va='top',    fontsize=8, alpha=0.6)
    fig.text(0.98, 0.98, STUDENT_NAME, ha='right', va='top',    fontsize=8, alpha=0.6)

def visualize_classifier(classifier, X, y, title=None, save_path=None):
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.3)

    classes = np.unique(y)
    for c in classes:
        pts = X[y == c]
        ax.scatter(pts[:, 0], pts[:, 1], label=f"class {int(c)}", s=18)

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    if title:
        ax.set_title(title)
    ax.legend()

    _add_watermarks(fig)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
