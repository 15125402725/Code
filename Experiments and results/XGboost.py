import xgboost as xgb
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
from matplotlib.gridspec import GridSpec

# Create directory to save images
save_dir = "model_evaluation_plots"
os.makedirs(save_dir, exist_ok=True)

# 1. Data loading
df = pd.read_csv('COUNT_SIS_selected_features.csv')
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

# 2. Data split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# 3. Basic XGBoost model
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)

model.fit(X_train, y_train)

# 5. Prediction
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# 6. Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
average_precision = average_precision_score(y_test, y_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")
print(f"Average Precision: {average_precision:.4f}")

# Generate timestamp for filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ====================== Separate Charts ======================

# 7.1 Confusion Matrix saved separately
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (After SMOTE)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
cm_path = os.path.join(save_dir, f"confusion_matrix_{timestamp}.png")
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.close()

# 7.2 Metric bar chart saved separately
plt.figure(figsize=(8, 5))
metrics = ['Accuracy', 'F1 Score', 'AUC-ROC', 'Avg Precision']
values = [accuracy, f1, roc_auc, average_precision]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
bars = plt.bar(metrics, values, color=colors)
plt.ylim(0, 1.1)
plt.title('Model Metrics Comparison (After SMOTE)')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}',
             ha='center', va='bottom')
metrics_path = os.path.join(save_dir, f"metrics_comparison_{timestamp}.png")
plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
plt.close()

# ====================== Combined Curve Chart ======================

# 7.3 Create combined ROC and PR curve chart
# 9.4 Combined ROC and PR curves
plt.figure(figsize=(8, 6))
plt.title('ROC & Precision-Recall Curves', fontsize=14, pad=20)

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, color='blue', lw=2,
         label=f'ROC (AUC = {roc_auc_score(y_test, y_proba):.3f})')

# PR curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)
plt.plot(recall, precision, color='red', linestyle='--', lw=2,
         label=f'PR (AP = {average_precision_score(y_test, y_proba):.3f})')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall / True Positive Rate', fontsize=12)
plt.ylabel('Precision / Positive Predictive Value', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(loc='lower right', fontsize=12, frameon=False)
combined_curve_path = os.path.join(save_dir, f"combined_curves_{timestamp}.png")
plt.savefig(combined_curve_path, dpi=300, bbox_inches='tight')
plt.close()

# ====================== Output Results ======================

# Print key threshold points
print("\nThresholds corresponding to key points:")
print(f"- When Recall=0.9, Precision={precision[recall >= 0.9][-1]:.2f}")
print(f"- When Precision=0.9, Recall={recall[precision >= 0.9][0]:.2f}")

# Print saved paths
print("\nCharts have been saved to the directory:")
print(f"- Confusion Matrix: {cm_path}")
print(f"- Metric Comparison: {metrics_path}")
print(f"- ROC/PR Combined Curve: {combined_curve_path}")
