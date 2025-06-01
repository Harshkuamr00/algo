from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
# Grid Search for SVM Classifier on Iris Dataset
# This code performs a grid search to find the best hyperparameters for an SVM classifier on the Iris dataset.
# Load the iris dataset

iris=datasets.load_iris()
# Check the values
df=pd.DataFrame(data=iris.data, columns=iris.feature_names)
df.to_csv('iris_dataset.csv', index=False)
x,y=iris.data,iris.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

para_grid={'C':[0.1,1,10,100], 'gamma':[0.01,0.1,10,100], 'kernel':['lenear','rbf','poly']}

# visulization of the dataset
plt.figure(figsize=(12, 8))
plt.scatter(x[:, 0], x[:, 1], c=y, cmap='viridis', edgecolor='k', s=100)
plt.title("Iris Dataset Visualization")
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.colorbar(label='Species')
plt.savefig('iris_dataset_visualization.png')
plt.show()
# Perform grid search to find the best hyperparameters
svm_classifier=SVC()
grid_search=GridSearchCV(svm_classifier, para_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train, y_train)
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
y_pred = grid_search.predict(x_test)

print(f"Best parameters: {best_params}")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
