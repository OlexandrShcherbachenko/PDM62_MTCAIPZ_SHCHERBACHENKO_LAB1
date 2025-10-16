
# LR_1_task_6.py
import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utilities import visualize_classifier

data = np.loadtxt('data_multivar_nb.txt', delimiter=',')
X, y = data[:, :-1], data[:, -1].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3, stratify=y)

svm = SVC(kernel='rbf', gamma='scale')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

nb = GaussianNB().fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

def report(name, y_true, y_pred):
    print(name, "accuracy", accuracy_score(y_true,y_pred))
    print(name, "precision", precision_score(y_true,y_pred,average='weighted'))
    print(name, "recall", recall_score(y_true,y_pred,average='weighted'))
    print(name, "f1", f1_score(y_true,y_pred,average='weighted'))

report("SVM", y_test, y_pred_svm)
report("NaiveBayes", y_test, y_pred_nb)

visualize_classifier(svm, X_test, y_test, title="SVM — test split", save_path="LR_1_task_6_svm.png")
visualize_classifier(nb, X_test, y_test, title="Naive Bayes — test split", save_path="LR_1_task_6_nb.png")
