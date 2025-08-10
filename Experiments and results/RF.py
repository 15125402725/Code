from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (average_precision_score, accuracy_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, precision_recall_curve)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from imblearn.over_sampling import SMOTE

# Create a directory to save plots
save_dir = "random_forest_evaluation_plots"
os.makedirs(save_dir, exist_ok=True)

# 1. Load data
df = pd.read_csv('COUNT_SIS_selected_features.csv')
X = df.iloc[:, 1:].values  # Feature columns (all columns except the first)
y = df.iloc[:, 0].values   # Target variable (first column)

# 2. Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y  # Stratify ensures the class distribution is similar in train and test sets
)

# 3. Apply SMOTE for handling class imbalance (optional)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# 4. Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

# 5. Train the model
model.fit(X_train_smote, y_train_smote)

# 6. Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

# 7. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
avg_precision = average_precision_score(y_test, y_prob)

# Print the evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print(f"ROC AUC: {roc_auc}")
print(f"Average Precision: {avg_precision}")

# 8. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
confusion_matrix_path = os.path.join(save_dir, "confusion_matrix.png")
plt.savefig(confusion_matrix_path)
plt.close()

# 9. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label="ROC Curve (AUC = %0.2f)" % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
roc_curve_path = os.path.join(save_dir, "roc_curve.png")
plt.savefig(roc_curve_path)
plt.close()

# 10. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='green', label="Precision-Recall Curve (AP = %0.2f)" % avg_precision)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
pr_curve_path = os.path.join(save_dir, "pr_curve.png")
plt.savefig(pr_curve_path)
plt.close()

# 11. Combined ROC/PR curve (for comparison)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='blue', label="ROC Curve (AUC = %0.2f)" % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.subplot(1, 2, 2)
plt.plot(recall, precision, color='green', label="Precision-Recall Curve (AP = %0.2f)" % avg_precision)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")

combined_curve_path = os.path.join(save_dir, "combined_roc_pr_curve.png")
plt.savefig(combined_curve_path)
plt.close()

print(f"- ROC/PR combined curve saved to: {combined_curve_path}")
