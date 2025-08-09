#sklearn.feature_selection.SelectKBest：用于特征选择，基于统计方法选择最相关的特征。
#SelectKBest 是 sklearn（scikit-learn）中的一个类，用于选择具有最高得分的特征。你使用了 f_classif（ANOVA F检验）作为选择标准。
#k=100 指定选择前100个特征。

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

# 读取方差过滤后的数据
filtered_data = pd.read_csv('PRAD_LUAD_filtered_gene_data.csv')

# 提取目标变量和基因表达数据
X = filtered_data.drop(columns=['y'])  # 基因表达数据
y = filtered_data['y']  # 目标变量

# 使用 SIS 方法（SelectKBest）进行特征选择，基于ANOVA F-value
selector_sis = SelectKBest(score_func=f_classif, k=100)  # 选择前500个最相关的特征
X_selected = selector_sis.fit_transform(X, y)

# 获取选中的特征名称
selected_features_sis = X.columns[selector_sis.get_support()]

# 重新构建筛选后的数据框（包括目标变量 'y' 和筛选后的基因数据）
filtered_data_sis = pd.DataFrame(X_selected, columns=selected_features_sis)
filtered_data_sis['y'] = y

# 调整'y'为第一列
cols = ['y'] + [col for col in filtered_data_sis.columns if col != 'y']
filtered_data_sis = filtered_data_sis[cols]

# 保存SIS筛选后的数据到新的CSV文件
filtered_data_sis.to_csv('PRAD_LUAD_filtered_gene_data_SIS.csv', index=False)
print("第二阶段 SIS 筛选后的数据已保存为 'PRAD_LUAD_filtered_gene_data_SIS.csv'，y为第一列。")