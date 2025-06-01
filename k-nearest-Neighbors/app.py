from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



# Load the iris dataset
iris=load_iris()
x,y=iris.data, iris.target

# make dataframe
df=pd.DataFrame(data=x, columns=iris.feature_names)
df['target']=y
df.to_csv('iris_dataset.csv', index=False)

# split the dataset into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#create a knn classifier
knn_classifier=KNeighborsClassifier(n_neighbors=3)
# fit the model
knn_classifier.fit(x_train,y_train)
# predict the labels for the test set
y_pred=knn_classifier.predict(x_test)

# calculate the accuracy
accuracy =accuracy_score(y_test,y_pred)
print(f"Accuracy of KNN classifier: {accuracy}")

# Visualize the dataset
plt.figure(figsize=(12, 8))
plt.scatter(x[:,0],x[:,1],c=y,cmap='viridis', edgecolor='k', s=100)
plt.title("Iris Dataset Visualization")
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.colorbar(label='Species')
plt.savefig('iris_dataset_visualization.png')
plt.show()

