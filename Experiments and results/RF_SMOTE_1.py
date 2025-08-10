from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (average_precision_score, accuracy_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, precision_recall_curve)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime
from imblearn.over_sampling import SMOTE

# Create a directory to save images
save_dir = "random_forest_evaluation_plots/RFSMOTE"
os.makedirs(save_dir, exist_ok=True)

# 1. Data Load
df = pd.read_csv("COUNT_SIS_selected_features.csv")
X = df.iloc[:, 1:].values  # Feature columns start from the second column.
y = df.iloc[:, 0].values   # The label column is the first column.

# 2. Data Splitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# ========== Cross-validation section ==========
print("Five-fold cross-validation (training set + SMOTE) for various metrics:")
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []
fold_f1_scores = []
fold_roc_aucs = []
fold_avg_precisions = []

for fold, (train_index, val_index) in enumerate(kf.split(X_train, y_train)):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    # Perform SMOTE within each fold.
    smote_cv = SMOTE(random_state=42)
    X_train_smote_fold, y_train_smote_fold = smote_cv.fit_resample(X_train_fold, y_train_fold)
    model_cv = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    )
    model_cv.fit(X_train_smote_fold, y_train_smote_fold)
    y_pred_cv = model_cv.predict(X_val_fold)
    y_proba_cv = model_cv.predict_proba(X_val_fold)[:, 1]
    fold_accuracies.append(accuracy_score(y_val_fold, y_pred_cv))
    fold_f1_scores.append(f1_score(y_val_fold, y_pred_cv))
    fold_roc_aucs.append(roc_auc_score(y_val_fold, y_proba_cv))
    fold_avg_precisions.append(average_precision_score(y_val_fold, y_proba_cv))
    print(f"Fold {fold+1}: Acc={fold_accuracies[-1]:.4f} | F1={fold_f1_scores[-1]:.4f} | "
          f"AUC={fold_roc_aucs[-1]:.4f} | AP={fold_avg_precisions[-1]:.4f}")

print("Five-fold cross-validation average：")
print(f"Accuracy: {np.mean(fold_accuracies):.4f}")
print(f"F1 Score: {np.mean(fold_f1_scores):.4f}")
print(f"AUC-ROC: {np.mean(fold_roc_aucs):.4f}")
print(f"Average Precision: {np.mean(fold_avg_precisions):.4f}")

# ========== Test set section ==========
# 3. Use SMOTE to handle class imbalance (process only the training set).
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# 4. Random forest model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train_smote, y_train_smote)

# 5. Prediction
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# 6. Evaluation Indicators
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
average_precision = average_precision_score(y_test, y_proba)

print("\n Test set evaluation：")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")
print(f"Average Precision: {average_precision:.4f}")

# Generate a timestamp for use in a filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ====================== Independent chart=====================

# 7.1 Confusion matrix saved separately.
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Random Forest with SMOTE)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
cm_path = os.path.join(save_dir, f"confusion_matrix_{timestamp}.png")
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.close()

# 7.2 Save the bar graph separately.
plt.figure(figsize=(8, 5))
metrics = ['Accuracy', 'F1 Score', 'AUC-ROC', 'Avg Precision']
values = [accuracy, f1, roc_auc, average_precision]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
bars = plt.bar(metrics, values, color=colors)
plt.ylim(0, 1.1)
plt.title('Random Forest Metrics Comparison (with SMOTE)')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}',
             ha='center', va='bottom')
metrics_path = os.path.join(save_dir, f"metrics_comparison_{timestamp}.png")
plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
plt.close()

# ====================== joint plot ======================

# 7.3 Create a combined plot of ROC and PR curves
plt.figure(figsize=(8, 6))
plt.title('ROC & Precision-Recall Curves (RF_SMOTE)', fontsize=14, pad=20)

# ROCcurve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, color='blue', lw=2,
         label=f'ROC (AUC = {roc_auc_score(y_test, y_proba):.3f})')

# PRcurve
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

# ====================== Output Result ======================

print("\nThresholds Corresponding key points：")
print(f"- when Recall=0.9，Precision={precision[recall >= 0.9][-1]:.2f}")
print(f"- when Precision=0.9，Recall={recall[precision >= 0.9][0]:.2f}")

print("\n The chart has been saved to the directory.:")
print(f"- confusion matrix: {cm_path}")
print(f"- indicator comparison: {metrics_path}")
print(f"- ROC/PR combined curve: {combined_curve_path}")
