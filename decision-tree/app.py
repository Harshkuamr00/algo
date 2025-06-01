from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd


# load the database from iris
iris=load_iris()

# check the values
df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['target'] = iris.target
#print(df.head())
df.to_csv('iris.data.csv', index=False)
# split the dataset into test and train
x, y=iris.data, iris.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

classifier=DecisionTreeClassifier(random_state=42)
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

# accuracy score
accuracy=accuracy_score(y_test,y_pred)

print(f'''Accuracy Score : {accuracy}''')

# plotting the decision tree
plt.figure(figsize=(12,8))
plot_tree(
    classifier,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True)
plt.title("Decision Tree")
plt.show()