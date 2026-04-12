import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib

# Load the data
df = pd.read_csv("stress_dataset.csv")

# Clean and rename column if needed
df.dropna(inplace=True)
df.rename(columns={"Time(sec)": "Time_sec"}, inplace=True)

# Features and labels
X = df[["HR", "respr"]]  # You can also include 'Time_sec' if it's meaningful
y = df["Label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
lr_model = LogisticRegression(max_iter=1000)

# Fit models
rf_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)

# Predict on the test set
rf_preds = rf_model.predict(X_test)
lr_preds = lr_model.predict(X_test)

# Confusion Matrix for Random Forest
rf_cm = confusion_matrix(y_test, rf_preds)

# Confusion Matrix for Logistic Regression
lr_cm = confusion_matrix(y_test, lr_preds)

# Plot Heatmap for Random Forest Confusion Matrix
plt.figure(figsize=(10, 6))
sns.heatmap(
    rf_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Stressed", "Stressed"],
    yticklabels=["Not Stressed", "Stressed"],
)
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Plot Heatmap for Logistic Regression Confusion Matrix
plt.figure(figsize=(10, 6))
sns.heatmap(
    lr_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Stressed", "Stressed"],
    yticklabels=["Not Stressed", "Stressed"],
)
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
