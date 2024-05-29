import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
df = pd.read_csv("loan_data.csv")

# Display first few rows
print(df.head())

# Display information about the dataset
print(df.info())

# Statistical summary
print(df.describe())

# Pairplot
sns.pairplot(data=df)
plt.show()

# Plot histograms for FICO score based on credit.policy
plt.figure(figsize=(10, 6))
df[df['credit.policy'] == 1]['fico'].hist(alpha=0.5, color='blue', bins=30, label='Credit Policy = 1', edgecolor='black')
df[df['credit.policy'] == 0]['fico'].hist(alpha=0.5, color='red', bins=30, label='Credit Policy = 0', edgecolor='black')
plt.legend()
plt.show()

# Plot histograms for FICO score based on not.fully.paid
plt.figure(figsize=(10, 6))
df[df['not.fully.paid'] == 1]['fico'].hist(alpha=0.5, color='blue', bins=30, label='not.fully.paid = 1', edgecolor='black')
df[df['not.fully.paid'] == 0]['fico'].hist(alpha=0.5, color='red', bins=30, label='not.fully.paid = 0', edgecolor='black')
plt.legend()
plt.show()

# Countplot for purpose based on not.fully.paid
plt.figure(figsize=(14, 8))
sns.countplot(data=df, x="purpose", hue="not.fully.paid")
plt.show()

# Jointplot for FICO score vs Interest rate
sns.jointplot(data=df, x='fico', y='int.rate', kind='scatter')
plt.show()

# Lmplots
sns.lmplot(data=df, x='fico', y='int.rate', hue='not.fully.paid', col='credit.policy', height=6, aspect=1)
plt.show()

# One-hot encoding of categorical features
cat_feats = ['purpose']
final_data = pd.get_dummies(df, columns=cat_feats, drop_first=True)

# Split the data into training and testing sets
X = final_data.drop('not.fully.paid', axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Decision Tree Classifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

predictions = dtree.predict(X_test)
print("Decision Tree Classifier Report:")
print(classification_report(y_test, predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Grid Search for best parameters
np.random.seed(101)  # Setting a seed for reproducibility
param_grid = {
    'max_depth': [None] + list(np.random.randint(10, 51, size=5)),
    'min_samples_split': np.random.randint(2, 21, size=3).tolist(),
    'min_samples_leaf': np.random.randint(1, 11, size=3).tolist(),
    'max_features': [None, 'auto', 'sqrt', 'log2']
}

grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best parameters found: ", grid_search.best_params_)

best_dtree = grid_search.best_estimator_
predictions = best_dtree.predict(X_test)
print("Grid Search Decision Tree Classifier Report:")
print(classification_report(y_test, predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=101)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
print("Random Forest Classifier Report:")
print(classification_report(y_test, rf_predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))
