import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================================
# SETUP: Read your dataset directly from your GitHub!
# ==========================================================
# This reads the file straight from your repository so you don't get the FileNotFoundError
url = 'https://raw.githubusercontent.com/krithiika-s/Predictive-Analytics-Lab-Exam-2/main/Lab_Exam_binary_classification_dataset.csv'
df = pd.read_csv(url)

# Clean the data by dropping rows where the Target is missing
df = df.dropna(subset=['Target'])

# Define Features (X) and Target (y)
X = df[['Feature1', 'Feature2']]
y = df['Target']

# Convert 'Yes'/'No' to numeric values (1 and 0)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ==========================================================
# Task 1: Exploratory Data Analysis (EDA) [7 Marks]
# ==========================================================
print("--- EDA: Dataset Info ---")
print(df.info())
print("\n--- EDA: Missing Values ---")
print(df.isnull().sum())

# Plotting feature distribution
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Feature1', y='Feature2', hue='Target', palette='Set1')
plt.title('EDA: Scatter Plot of Features')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# ==========================================================
# Task 2: Build an appropriate classification model [8 Marks]
# ==========================================================
# Split the dataset (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Initialize and train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ==========================================================
# Task 3: Plot the decision boundary [4 Marks]
# ==========================================================
# Define grid bounds based on training data
x_min, x_max = X_train['Feature1'].min() - 1, X_train['Feature1'].max() + 1
y_min, y_max = X_train['Feature2'].min() - 1, X_train['Feature2'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                     np.arange(y_min, y_max, 0.05))

# Predict across the grid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotting
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='Set1')
sns.scatterplot(x=X_train['Feature1'], y=X_train['Feature2'], hue=le.inverse_transform(y_train), palette='Set1', edgecolor='k')
plt.title('Decision Boundary of Random Forest Classifier')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# ==========================================================
# Task 4: Evaluate and report model performance [6 Marks]
# ==========================================================
y_pred = model.predict(X_test)

print("\n--- Model Evaluation ---")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")

print("\nClassification Report:")
# Convert encoded labels back to original for clear reading
target_names = le.classes_
print(classification_report(y_test, y_pred, target_names=target_names))

print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Visualize Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
