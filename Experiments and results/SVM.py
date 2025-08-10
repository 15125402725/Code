from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (average_precision_score, accuracy_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, precision_recall_curve)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import numpy as np

# Set the font for displaying Chinese (ensure the system has Chinese fonts)
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows system Chinese display
plt.rcParams['axes.unicode_minus'] = False  # Solve the issue of displaying negative signs

# Create a directory to save the plots
save_dir = "nystroem_svm_evaluation_plots"
os.makedirs(save_dir, exist_ok=True)

# 1. Data loading
df = pd.read_csv('COUNT_SIS_selected_features.csv')
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

# 2. Data splitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# 3. Data preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Automatically calculate the gamma value for RBF kernel (equivalent to gamma='scale')
gamma_value = 1 / (X_train_scaled.shape[1] * X_train_scaled.var())

# ========== Five-fold cross-validation ========== 
print("\n=== Five-fold cross-validation evaluation ===")
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []
fold_f1_scores = []
fold_roc_aucs = []
fold_avg_precisions = []

for fold, (train_index, val_index) in enumerate(kf.split(X_train_scaled, y_train)):
    X_train_fold, X_val_fold = X_train_scaled[train_index], X_train_scaled[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    # Build Nystroem kernel approximation
    nystroem = Nystroem(
        kernel='rbf',
        gamma=gamma_value,
        n_components=100,
        random_state=42
    )

    # Build SVM classifier
    svm = SGDClassifier(
        loss='hinge',
        penalty='l2',
        alpha=0.001,
        max_iter=1000,
        tol=1e-3,
        random_state=42,
        class_weight='balanced'
    )

    # Build pipeline
    model = make_pipeline(nystroem, svm)

    # Calibrate the model to get probability outputs
    calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=5)
    calibrated_model.fit(X_train_fold, y_train_fold)

    y_pred_cv = calibrated_model.predict(X_val_fold)
    y_proba_cv = calibrated_model.predict_proba(X_val_fold)[:, 1]

    # Calculate evaluation metrics
    fold_accuracies.append(accuracy_score(y_val_fold, y_pred_cv))
    fold_f1_scores.append(f1_score(y_val_fold, y_pred_cv))
    fold_roc_aucs.append(roc_auc_score(y_val_fold, y_proba_cv))
    fold_avg_precisions.append(average_precision_score(y_val_fold, y_proba_cv))

    print(f"Fold {fold + 1}: "
          f"Acc={fold_accuracies[-1]:.4f} | "
          f"F1={fold_f1_scores[-1]:.4f} | "
          f"AUC={fold_roc_aucs[-1]:.4f} | "
          f"AP={fold_avg_precisions[-1]:.4f}")

print("\nFive-fold cross-validation average results:")
print(f"Accuracy: {np.mean(fold_accuracies):.4f} (±{np.std(fold_accuracies):.4f})")
print(f"F1 Score: {np.mean(fold_f1_scores):.4f} (±{np.std(fold_f1_scores):.4f})")
print(f"AUC-ROC: {np.mean(fold_roc_aucs):.4f} (±{np.std(fold_roc_aucs):.4f})")
print(f"Average Precision: {np.mean(fold_avg_precisions):.4f} (±{np.std(fold_avg_precisions):.4f})")

# 5. Train the final model
print("\nTraining final model...")
# Build Nystroem kernel approximation
nystroem = Nystroem(
    kernel='rbf',
    gamma=gamma_value,
    n_components=100,
    random_state=42
)

# Build SVM classifier
svm = SGDClassifier(
    loss='hinge',
    penalty='l2',
    alpha=0.001,
    max_iter=1000,
    tol=1e-3,
    random_state=42,
    class_weight='balanced'
)

# Build pipeline
model = make_pipeline(nystroem, svm)

# Calibrate the model to get probability outputs
calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=5)
calibrated_model.fit(X_train_scaled, y_train)

# 6. Prediction
y_pred = calibrated_model.predict(X_test_scaled)
y_proba = calibrated_model.predict_proba(X_test_scaled)[:, 1]

# 7. Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
average_precision = average_precision_score(y_test, y_proba)

print("\nTest set evaluation results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"Average Precision: {average_precision:.4f}")

# Generate timestamp for file name
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ====================== Visualization ======================

# 8.1 Confusion matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['True Negative', 'True Positive'])
plt.title('Confusion Matrix (Nystroem SVM)')
cm_path = os.path.join(save_dir, f"confusion_matrix_{timestamp}.png")
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.close()

# 8.2 Metrics bar chart
plt.figure(figsize=(8, 5))
metrics = ['Accuracy', 'F1 Score', 'AUC-ROC', 'Average Precision']
values = [accuracy, f1, roc_auc, average_precision]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
bars = plt.bar(metrics, values, color=colors)
plt.ylim(0, 1.1)
plt.title('Nystroem SVM Model Evaluation Metrics Comparison')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height,
             f'{height:.3f}',
             ha='center', va='bottom')
metrics_path = os.path.join(save_dir, f"metrics_comparison_{timestamp}.png")
plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
plt.close()

# 8.3 ROC and PR curve combined plot
plt.figure(figsize=(8, 6))
plt.title('ROC and Precision-Recall Curves (Nystroem SVM)', fontsize=14, pad=20)

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, color='blue', lw=2,
         label=f'ROC Curve (AUC = {roc_auc:.3f})')

# PR curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)
plt.plot(recall, precision, color='red', linestyle='--', lw=2,
         label=f'PR Curve (AP = {average_precision:.3f})')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(loc='lower right', fontsize=12, frameon=False)
combined_curve_path = os.path.join(save_dir, f"combined_curves_{timestamp}.png")
plt.savefig(combined_curve_path, dpi=300, bbox_inches='tight')
plt.close()

# ====================== Output results ======================

# Print key threshold points
print("\nKey threshold points analysis:")
print(f"- When recall = 0.9, precision = {precision[recall >= 0.9][-1]:.2f}")
print(f"- When precision = 0.9, recall = {recall[precision >= 0.9][0]:.2f}")

# Print saved paths
print("\nVisualization charts have been saved to the following paths:")
print(f"- Confusion Matrix: {cm_path}")
print(f"- Metrics Comparison: {metrics_path}")
print(f"- ROC/PR Combined Curve: {combined_curve_path}")
