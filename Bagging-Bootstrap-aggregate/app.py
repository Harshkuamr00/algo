import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits

# Load dataset
digit = load_digits()
x, y = digit.data, digit.target

# Create DataFrame with generated feature names
df = pd.DataFrame(data=x, columns=[f'feature_{i}' for i in range(x.shape[1])])  # Fixed feature names
df.to_csv('digits_dataset.csv', index=False)  # Save DataFrame to CSV
# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize and train classifier
base_classifier = DecisionTreeClassifier(random_state=42)
bagging_classifier = BaggingClassifier(base_classifier, n_estimators=10, random_state=42)
bagging_classifier.fit(x_train, y_train)

# Evaluate
y_pred = bagging_classifier.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Plot decision tree
plt.figure(figsize=(10, 6))
plot_tree(
    bagging_classifier.estimators_[0],
    feature_names=df.columns,
    class_names=[str(c) for c in digit.target_names],  # Convert to strings
    filled=True,
    rounded=True
)
plt.title("Decision Tree in Bagging Classifier")
plt.savefig('bagging_classifier_tree.png')
plt.show()
