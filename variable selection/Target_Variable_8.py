#pandas：用于数据读取和处理。
#sklearn.feature_selection.VarianceThreshold：用于特征选择，通过方差过滤法来选择具有较高方差的特征。

import pandas as pd
from sklearn.feature_selection import VarianceThreshold

# 读取数据
file_path = 'PRAD_LUAD.csv'
data = pd.read_csv(file_path)
# 提取基因表达数据部分
gene_data = data.drop(columns=['y'])
# 使用方差过滤法进行特征选择，设定方差阈值
selector = VarianceThreshold(threshold=0.01)  # 可以根据需要调整阈值
selected_gene_data = selector.fit_transform(gene_data)
# 获取选择后的特征名称
selected_features = gene_data.columns[selector.get_support()]
# 重新构建过滤后的数据框（包括目标变量 'y' 和筛选后的基因数据）
filtered_data = pd.DataFrame(selected_gene_data, columns=selected_features)
filtered_data['y'] = data['y']
# 保存过滤后的数据到新的 CSV 文件
filtered_data.to_csv('PRAD_LUAD_filtered_gene_data.csv', index=False)
print("过滤后的数据已保存为 'PRAD_LUAD_filtered_gene_data.csv'")
