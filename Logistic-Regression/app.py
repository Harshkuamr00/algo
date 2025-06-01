from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
x,y=make_classification(n_samples=1000,n_features=2,n_redundant=0,n_clusters_per_class=1,random_state=42)
# Create a DataFrame
df=pd.DataFrame(x, columns=['feature_1', 'feature_2'])
df['target'] = y
df.to_csv('logistic_regression_dataset.csv', index=False)

# split the dataset into train and test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
# Create a Logistic Regression model
model=LogisticRegression(random_state=42)
model.fit(x_train,y_train)

# Predict the target variable
y_pred=model.predict(x_test)

# Calculate accuracy
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
# Generate confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'''Confusion Matrix:
{conf_matrix}
Classification Report:
{classification_report(y_test, y_pred)}''')
# Plot the decision boundary
plt.figure(figsize=(10, 6))
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap='viridis', edgecolors='k', s=50, label='Test Data')
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap='coolwarm', edgecolors='k', s=50, label='Train Data')
# Create a grid to plot the decision boundary
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))
# Predict the class for each point in the grid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.title('Logistic Regression Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.savefig('logistic_regression_decision_boundary.png')
plt.show()
