import numpy as np
import pandas as pd
from sklearn.svm import SVC  # 改为导入SVM
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, average_precision_score)
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.utils import resample
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV  # 新增用于概率校准

# 1. 数据准备模块
class DataPreparer:
    def __init__(self, test_size=0.2, calib_size=0.3, random_state=42):
        self.test_size = test_size
        self.calib_size = calib_size
        self.random_state = random_state

    def load_and_split(self, data_path):
        df = pd.read_csv(data_path)
        X = df.iloc[:, 1:].values
        y = df.iloc[:, 0].values

        # 第一次分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=self.random_state)

        # 对训练集应用SMOTE
        smote = SMOTE(random_state=self.random_state)
        X_res, y_res = smote.fit_resample(X_train, y_train)

        # 第二次分割
        n = len(X_res)
        idx = np.random.RandomState(self.random_state).permutation(n)
        split_idx = n // 2

        X_train, X_cal = X_res[idx[:split_idx]], X_res[idx[split_idx:]]
        y_train, y_cal = y_res[idx[:split_idx]], y_res[idx[split_idx:]]

        return X_train, y_train, X_cal, y_cal, X_test, y_test

# 2. 模型训练模块（改为SVM）
class ModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        # 使用带概率校准的SVM
        self.model = CalibratedClassifierCV(
            SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=random_state),
            cv=5,
            method='sigmoid'
        )

    def cross_validate(self, X, y, n_splits=5):
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        metrics = []
        for train_idx, val_idx in kf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_val)
            y_proba = self.model.predict_proba(X_val)[:, 1]
            metrics.append({
                'accuracy': accuracy_score(y_val, y_pred),
                'f1': f1_score(y_val, y_pred),
                'roc_auc': roc_auc_score(y_val, y_proba),
                'gmean': geometric_mean_score(y_val, y_pred),
                'avg_precision': average_precision_score(y_val, y_proba)
            })
        return pd.DataFrame(metrics)

    def train_final_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self.model

# 3. 共形预测模块（保持不变）
class AdvancedConformalPredictor:
    def __init__(self, confidence_level=0.95, n_bootstrap=1000, random_state=42):
        assert 0 < confidence_level < 1, "confidence_level必须在0和1之间"
        self.alpha = 1 - confidence_level
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.q_hat = None

    def _calculate_nonconformity(self, model, X, y):
        probas = model.predict_proba(X)
        return 1 - probas[np.arange(len(y)), y]

    def predict_with_confidence(self, model, X_cal, y_cal, X_test):
        cal_scores = self._calculate_nonconformity(model, X_cal, y_cal)
        n = len(cal_scores)

        # 关键修复：安全计算k值并处理边界情况
        k = min(int(np.ceil((1 - self.alpha) * (n + 1))), n)
        self.q_hat = np.sort(cal_scores)[max(0, k - 1)]  # 确保索引非负

        test_probas = model.predict_proba(X_test)
        prediction_sets = [np.where(prob >= (1 - self.q_hat))[0] for prob in test_probas]

        # 计算覆盖率的置信区间
        coverage_samples = []
        for _ in range(self.n_bootstrap):
            boot_scores = resample(cal_scores, replace=True, random_state=self.random_state)
            q_boot = np.sort(boot_scores)[min(k - 1, len(boot_scores) - 1)]
            boot_sets = [np.where(prob >= (1 - q_boot))[0] for prob in test_probas]
            coverage_samples.append(np.mean([y_test[i] in boot_sets[i] for i in range(len(y_test))]))

        coverage_ci = (np.percentile(coverage_samples, 2.5), np.percentile(coverage_samples, 97.5))
        return prediction_sets, self.q_hat, coverage_ci

    def plot_error_distribution(self, model, X_cal, y_cal):
        cal_scores = self._calculate_nonconformity(model, X_cal, y_cal)

        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        sns.histplot(cal_scores, kde=True, bins=20)
        plt.axvline(self.q_hat, color='r', linestyle='--',
                    label=f'Critical value (α={self.alpha})')
        plt.title("Nonconformity Scores Distribution")
        plt.xlabel("1 - P(y_true)")
        plt.legend()

        plt.subplot(122)
        ecdf = np.sort(cal_scores)
        plt.plot(ecdf, np.linspace(0, 1, len(ecdf)), label='ECDF')
        plt.axhline(1 - self.alpha, color='g', linestyle=':', label='1-α level')
        plt.axvline(self.q_hat, color='r', linestyle='--')
        plt.title("Quantile Position")
        plt.xlabel("Nonconformity Score")
        plt.ylabel("Cumulative Probability")
        plt.legend()
        plt.tight_layout()
        plt.savefig("results/error_distribution.png")
        plt.close()

# 4. 可视化模块（保持不变）
class ResultVisualizer:
    def __init__(self, save_dir="results"):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

    def plot_metrics(self, metrics_df, filename="cv_metrics.png"):
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=metrics_df, orient='h')
        plt.title('Cross-Validation Metrics Distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close()

    def plot_roc_pr(self, y_true, y_proba, filename="roc_pr.png"):
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)

        plt.plot(recall, precision, '--', color='red', linewidth=2,
                 label=f'PR (AP = {pr_auc:.3f})')
        plt.plot(fpr, tpr, '-', color='blue', linewidth=2,
                 label=f'ROC (AUC = {roc_auc:.3f})')

        plt.xlabel('Recall / True Positive Rate', fontsize=12)
        plt.ylabel('Precision / Positive Predictive Value', fontsize=12)
        plt.title('ROC & Precision-Recall Curves (SVM_SCP)', fontsize=14)  # 修改标题
        plt.grid(True, alpha=0.3)
        plt.legend(loc='lower right', fontsize=10)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close()

    def plot_coverage_vs_set_size(self, confidence_levels, coverage_rates, avg_set_sizes,
                                  filename="coverage_vs_set_size.png"):
        plt.figure(figsize=(8, 6))
        fig, ax1 = plt.subplots()

        # Coverage rate
        color = 'tab:blue'
        ax1.set_xlabel('Confidence Level (1-α)')
        ax1.set_ylabel('Coverage Rate', color=color)
        ax1.plot(confidence_levels, coverage_rates, '-o', color=color, linewidth=2, label='Coverage Rate')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0.5, 1.0)
        ax1.set_xlim(min(confidence_levels), max(confidence_levels))
        ax1.grid(True, alpha=0.3)

        # Set size
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Average Set Size', color=color)
        ax2.plot(confidence_levels, avg_set_sizes, '--s', color=color, linewidth=2, label='Average Set Size')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('Coverage Rate vs. Prediction Set Size (SVM)')  # 修改标题
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close()

# 主执行流程
if __name__ == "__main__":
    # 初始化组件
    preparer = DataPreparer(test_size=0.2, calib_size=0.5, random_state=42)
    trainer = ModelTrainer(random_state=42)
    visualizer = ResultVisualizer()
    conformal = AdvancedConformalPredictor(confidence_level=0.95)

    # 数据准备
    try:
        X_train, y_train, X_cal, y_cal, X_test, y_test = preparer.load_and_split(
            "COUNT_SIS_selected_features.csv"
        )
        print(f"数据加载成功！训练集: {X_train.shape}, 校准集: {X_cal.shape}, 测试集: {X_test.shape}")
    except FileNotFoundError:
        print("错误：未找到数据文件！请检查文件路径")
        exit()

    # 模型训练与评估
    print("\n正在进行交叉验证...")
    cv_results = trainer.cross_validate(X_train, y_train)
    print("交叉验证结果：")
    print(cv_results.describe().loc[['mean', 'std']].T)
    visualizer.plot_metrics(cv_results)

    # 训练最终模型
    print("\n训练最终模型...")
    final_model = trainer.train_final_model(X_train, y_train)

    # 共形预测
    print("\n执行分裂共形预测...")
    test_sets, q_hat, coverage_ci = conformal.predict_with_confidence(
        final_model, X_cal, y_cal, X_test
    )

    # 可视化误差分布
    conformal.plot_error_distribution(final_model, X_cal, y_cal)

    # 结果分析
    coverage = np.mean([y_test[i] in test_sets[i] for i in range(len(y_test))])
    avg_set_size = np.mean([len(s) for s in test_sets])

    # 常规评估
    y_pred = final_model.predict(X_test)
    y_proba = final_model.predict_proba(X_test)[:, 1]

    print("\n=== 模型评估结果 ===")
    print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1分数: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"平均精度: {average_precision_score(y_test, y_proba):.4f}")
    print(f"G-Mean: {geometric_mean_score(y_test, y_pred):.4f}")

    print("\n=== 分裂共形预测结果 ===")
    print(f"实际覆盖率: {coverage:.4f} (目标 ≥ {1 - conformal.alpha:.0%})")
    print(f"覆盖率95%置信区间: [{coverage_ci[0]:.4f}, {coverage_ci[1]:.4f}]")
    print(f"平均预测集大小: {avg_set_size:.4f}")

    # 测试不同置信水平
    print("\n测试不同置信水平下的覆盖率和预测集大小...")
    confidence_levels = np.linspace(0.5, 0.99, 10)
    coverage_rates = []
    avg_set_sizes = []

    for cl in confidence_levels:
        temp_conformal = AdvancedConformalPredictor(confidence_level=cl)
        temp_sets, _, _ = temp_conformal.predict_with_confidence(final_model, X_cal, y_cal, X_test)
        coverage_rates.append(np.mean([y_test[i] in temp_sets[i] for i in range(len(y_test))]))
        avg_set_sizes.append(np.mean([len(s) for s in temp_sets]))

    # 可视化
    #visualizer.plot_coverage_vs_set_size(confidence_levels, coverage_rates, avg_set_sizes)
    #print("\n所有可视化结果已保存至 results/ 目录")