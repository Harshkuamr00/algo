import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

# Example true and predicted labels
y_true = [0, 1, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)
labels = np.unique(y_true + y_pred)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()