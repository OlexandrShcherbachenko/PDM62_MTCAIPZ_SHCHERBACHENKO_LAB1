
# LR_1_task_4.py
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from utilities import visualize_classifier

data = np.loadtxt('data_multivar_nb.txt', delimiter=',')
X, y = data[:, :-1], data[:, -1].astype(int)

classifier = GaussianNB()
classifier.fit(X, y)
y_pred = classifier.predict(X)
accuracy = 100.0 * (y == y_pred).sum() / X.shape[0]
print("Accuracy =", round(accuracy, 2), "%")
visualize_classifier(classifier, X, y, title="Naive Bayes — full data", save_path="LR_1_task_4_full.png")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3, stratify=y)
classifier_new = GaussianNB()
classifier_new.fit(X_train, y_train)
y_test_pred = classifier_new.predict(X_test)
accuracy2 = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0]
print("Accuracy test =", round(accuracy2, 2), "%")
visualize_classifier(classifier_new, X_test, y_test, title="Naive Bayes — test split", save_path="LR_1_task_4_test.png")
