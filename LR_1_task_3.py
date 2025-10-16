
# LR_1_task_3.py
import numpy as np
from sklearn import linear_model
from utilities import visualize_classifier

X = np.array([[3.1, 7.2], [4.0, 6.7], [2.9, 8.0], 
              [5.1, 4.5], [6.0, 5.0], [5.6, 5.0], 
              [3.3, 0.4], [3.9, 0.9], [2.8, 1.0],
              [0.5, 3.4], [1.0, 4.0], [0.6, 4.9]])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

classifier = linear_model.LogisticRegression(solver='liblinear', C=1.0, multi_class='ovr')
classifier.fit(X, y)

visualize_classifier(classifier, X, y, title="Logistic Regression — decision regions", 
                     save_path="LR_1_task_3_decision.png")
print("Saved: LR_1_task_3_decision.png")
