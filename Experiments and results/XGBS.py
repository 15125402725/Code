import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (average_precision_score, accuracy_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, precision_recall_curve)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from imblearn.over_sampling import SMOTE
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建保存图片的目录
save_dir = "model_evaluation_plots"
os.makedirs(save_dir, exist_ok=True)

# 1. 数据加载
df = pd.read_csv('COUNT_SIS_selected_features.csv')
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

# 2. 数据分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# 2.1 应用SMOTE处理不平衡数据
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# ========== 五折交叉验证 ==========
print("\n=== 五折交叉验证评估 ===")
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []
fold_f1_scores = []
fold_roc_aucs = []
fold_avg_precisions = []

for fold, (train_index, val_index) in enumerate(kf.split(X_train_res, y_train_res)):
    X_train_fold, X_val_fold = X_train_res[train_index], X_train_res[val_index]
    y_train_fold, y_val_fold = y_train_res[train_index], y_train_res[val_index]

    # 定义XGBoost模型（与最终模型相同配置）
    model_cv = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )

    # 训练模型
    model_cv.fit(X_train_fold, y_train_fold)

    # 预测
    y_pred_cv = model_cv.predict(X_val_fold)
    y_proba_cv = model_cv.predict_proba(X_val_fold)[:, 1]

    # 计算评估指标
    fold_accuracies.append(accuracy_score(y_val_fold, y_pred_cv))
    fold_f1_scores.append(f1_score(y_val_fold, y_pred_cv))
    fold_roc_aucs.append(roc_auc_score(y_val_fold, y_proba_cv))
    fold_avg_precisions.append(average_precision_score(y_val_fold, y_proba_cv))

    print(f"Fold {fold + 1}: "
          f"Acc={fold_accuracies[-1]:.4f} | "
          f"F1={fold_f1_scores[-1]:.4f} | "
          f"AUC={fold_roc_aucs[-1]:.4f} | "
          f"AP={fold_avg_precisions[-1]:.4f}")

print("\n五折交叉验证平均结果:")
print(f"Accuracy: {np.mean(fold_accuracies):.4f} (±{np.std(fold_accuracies):.4f})")
print(f"F1 Score: {np.mean(fold_f1_scores):.4f} (±{np.std(fold_f1_scores):.4f})")
print(f"AUC-ROC: {np.mean(fold_roc_aucs):.4f} (±{np.std(fold_roc_aucs):.4f})")
print(f"Average Precision: {np.mean(fold_avg_precisions):.4f} (±{np.std(fold_avg_precisions):.4f})")

# 3. 训练最终模型
print("\n训练最终模型...")
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)

# 4. 训练模型
model.fit(X_train_res, y_train_res)

# 5. 预测
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# 6. 评估指标
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
average_precision = average_precision_score(y_test, y_proba)

print("\n测试集评估结果:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")
print(f"Average Precision: {average_precision:.4f}")

# 生成时间戳用于文件名
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ====================== 独立图表 ======================

# 7.1 混淆矩阵单独保存
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (After SMOTE)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
cm_path = os.path.join(save_dir, f"confusion_matrix_{timestamp}.png")
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.close()

# 7.2 指标条形图单独保存
plt.figure(figsize=(8, 5))
metrics = ['Accuracy', 'F1 Score', 'AUC-ROC', 'Avg Precision']
values = [accuracy, f1, roc_auc, average_precision]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
bars = plt.bar(metrics, values, color=colors)
plt.ylim(0, 1.1)
plt.title('Model Metrics Comparison (After SMOTE)')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height,
             f'{height:.3f}',
             ha='center', va='bottom')
metrics_path = os.path.join(save_dir, f"metrics_comparison_{timestamp}.png")
plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
plt.close()

# ====================== 单图绘制ROC和PR曲线 ======================
plt.figure(figsize=(8, 6))
plt.title('ROC & Precision-Recall Curves(XGBoost-SCP)', fontsize=14, pad=20)

# 绘制ROC曲线（蓝色实线）
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, color='blue', lw=2,
         label=f'ROC (AUC = {roc_auc:.3f})')

# 绘制PR曲线（红色虚线）
precision, recall, _ = precision_recall_curve(y_test, y_proba)
plt.plot(recall, precision, color='red', linestyle='--', lw=2,
         label=f'PR (AP = {average_precision:.3f})')

# 公共设置
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
plt.xlabel('Recall / True Positive Rate', fontsize=12)
plt.ylabel('Precision / Positive Predictive Value', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(loc='lower right', fontsize=12, frameon=False)

# 修正保存路径变量名（原curves_path改为combined_curve_path）
combined_curve_path = os.path.join(save_dir, f"xgboost_combined_curve_{timestamp}.png")
plt.savefig(combined_curve_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# ====================== 输出结果 ======================
print("\n图表已保存至目录:")
print(f"- 混淆矩阵: {cm_path}")
print(f"- 指标对比: {metrics_path}")
print(f"- 综合曲线图: {combined_curve_path}")