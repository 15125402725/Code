from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
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
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_imb_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

# 设置中文显示（确保系统有中文字体）
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建保存图片的目录
save_dir = "nystroem_svm_evaluation_plots"
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

# 3. 数据预处理 - 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. 自动计算RBF核的gamma值（替代gamma='scale'）
gamma_value = 1 / (X_train_scaled.shape[1] * X_train_scaled.var())

# ========== 交叉验证部分 ==========
print("\n五折交叉验证（训练集+SMOTE）各项指标：")
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []
fold_f1_scores = []
fold_roc_aucs = []
fold_avg_precisions = []

for fold, (train_index, val_index) in enumerate(kf.split(X_train_scaled, y_train)):
    X_train_fold, X_val_fold = X_train_scaled[train_index], X_train_scaled[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    # 每折内做SMOTE
    smote_cv = SMOTE(random_state=42)
    X_train_smote_fold, y_train_smote_fold = smote_cv.fit_resample(X_train_fold, y_train_fold)

    # 构建Nystroem SVM模型
    model_cv = make_imb_pipeline(
        Nystroem(
            kernel='rbf',
            gamma=gamma_value,
            n_components=100,
            random_state=42
        ),
        SGDClassifier(
            loss='hinge',
            penalty='l2',
            alpha=0.001,
            max_iter=1000,
            tol=1e-3,
            random_state=42,
            class_weight='balanced'
        )
    )

    # 包装校准模型以获得概率输出
    calibrated_model_cv = CalibratedClassifierCV(model_cv, method='sigmoid', cv=5)
    calibrated_model_cv.fit(X_train_smote_fold, y_train_smote_fold)

    y_pred_cv = calibrated_model_cv.predict(X_val_fold)
    y_proba_cv = calibrated_model_cv.predict_proba(X_val_fold)[:, 1]

    fold_accuracies.append(accuracy_score(y_val_fold, y_pred_cv))
    fold_f1_scores.append(f1_score(y_val_fold, y_pred_cv))
    fold_roc_aucs.append(roc_auc_score(y_val_fold, y_proba_cv))
    fold_avg_precisions.append(average_precision_score(y_val_fold, y_proba_cv))

    print(f"Fold {fold + 1}: Acc={fold_accuracies[-1]:.4f} | F1={fold_f1_scores[-1]:.4f} | "
          f"AUC={fold_roc_aucs[-1]:.4f} | AP={fold_avg_precisions[-1]:.4f}")

print("\n五折交叉验证平均：")
print(f"Accuracy: {np.mean(fold_accuracies):.4f} (±{np.std(fold_accuracies):.4f})")
print(f"F1 Score: {np.mean(fold_f1_scores):.4f} (±{np.std(fold_f1_scores):.4f})")
print(f"AUC-ROC: {np.mean(fold_roc_aucs):.4f} (±{np.std(fold_roc_aucs):.4f})")
print(f"Average Precision: {np.mean(fold_avg_precisions):.4f} (±{np.std(fold_avg_precisions):.4f})")

# 5. 构建完整管道（包含SMOTE、Nystroem和SVM）用于最终模型
model = make_imb_pipeline(
    SMOTE(random_state=42),  # SMOTE过采样
    Nystroem(
        kernel='rbf',
        gamma=gamma_value,  # 使用计算出的gamma值
        n_components=100,
        random_state=42
    ),
    SGDClassifier(
        loss='hinge',
        penalty='l2',
        alpha=0.001,
        max_iter=1000,
        tol=1e-3,
        random_state=42,
        class_weight='balanced'
    )
)

# 包装校准模型以获得概率输出
calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=5)
calibrated_model.fit(X_train_scaled, y_train)  # 注意：SMOTE会在交叉验证的每个fold内部应用

# 6. 预测
y_pred = calibrated_model.predict(X_test_scaled)
y_proba = calibrated_model.predict_proba(X_test_scaled)[:, 1]

# 7. 评估指标
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
average_precision = average_precision_score(y_test, y_proba)

print("\n模型评估结果（使用SMOTE+Nystroem+SVM）：")
print(f"准确率(Accuracy): {accuracy:.4f}")
print(f"F1分数(F1 Score): {f1:.4f}")
print(f"ROC曲线下面积(AUC-ROC): {roc_auc:.4f}")
print(f"平均精确率(Average Precision): {average_precision:.4f}")

# 生成时间戳用于文件名
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ====================== 可视化图表 ======================

# 8.1 混淆矩阵
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['预测负类', '预测正类'],
            yticklabels=['真实负类', '真实正类'])
plt.title('混淆矩阵 (Nystroem SVM with SMOTE)')
cm_path = os.path.join(save_dir, f"confusion_matrix_{timestamp}.png")
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.close()

# 8.2 指标条形图
plt.figure(figsize=(8, 5))
metrics = ['准确率', 'F1分数', 'AUC-ROC', '平均精确率']
values = [accuracy, f1, roc_auc, average_precision]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
bars = plt.bar(metrics, values, color=colors)
plt.ylim(0, 1.1)
plt.title('Nystroem SVM 模型评估指标对比 (使用SMOTE)')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height,
             f'{height:.3f}',
             ha='center', va='bottom')
metrics_path = os.path.join(save_dir, f"metrics_comparison_{timestamp}.png")
plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
plt.close()

# 8.3 ROC和PR曲线的联合图
plt.figure(figsize=(8, 6))
plt.title('ROC & Precision-Recall Curves(SMOTE_SVM)', fontsize=14, pad=20)

# ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, color='blue', lw=2,
         label=f'ROC (AUC = {roc_auc:.3f})')

# PR曲线
precision, recall, _ = precision_recall_curve(y_test, y_proba)
plt.plot(recall, precision, color='red', linestyle='--', lw=2,
         label=f'PR (AP = {average_precision:.3f})')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall / True Positive Rate', fontsize=12)
plt.ylabel('Precision / Positive Predictive Value', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(loc='lower right', fontsize=12, frameon=False)
combined_curve_path = os.path.join(save_dir, f"combined_curves_{timestamp}.png")
plt.savefig(combined_curve_path, dpi=300, bbox_inches='tight')
plt.close()

# ====================== 输出结果 ======================

# 打印关键阈值点
print("\n关键阈值点分析：")
print(f"- 当召回率=0.9时，精确率={precision[recall >= 0.9][-1]:.2f}")
print(f"- 当精确率=0.9时，召回率={recall[precision >= 0.9][0]:.2f}")

# 打印保存路径
print("\n可视化图表已保存至以下路径:")
print(f"- 混淆矩阵: {cm_path}")
print(f"- 指标对比图: {metrics_path}")
print(f"- ROC/PR联合曲线: {combined_curve_path}")