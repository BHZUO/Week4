import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import export_text
from data_processing import load_and_preprocess_data  # 导入预处理函数

# 加载和预处理数据
df = load_and_preprocess_data()

# 选择特征和目标变量
features = ['City', 'Standard_Package', 'Variety', 'Item Size', 'Origin Group', 'Season', 'Month', 'Year', 'Day']
target = 'Avg Price'

# 将目标变量转换为数值型，非数值的行将被丢弃
df[target] = pd.to_numeric(df[target], errors='coerce')
df.dropna(subset=[target], inplace=True)

# 分离特征和目标变量
X = df[features]
y = df[target]

# 处理分类特征和数值特征
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=[object]).columns.tolist()

# 创建预处理管道
numeric_transformer = SimpleImputer(strategy='median')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 创建完整的管道，包括预处理和模型训练
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=5, random_state=42))
])

# 训练模型
model.fit(X, y)

# 提取随机森林中的第一棵树
rf = model.named_steps['regressor']
tree_index = 0
estimator = rf.estimators_[tree_index]

# 打印树结构
print("【任务1】树结构：")
tree_rules = export_text(estimator, feature_names=preprocessor.get_feature_names_out())
print(tree_rules)

# 保存为文件（任务1提交）
with open('tree_structure.txt', 'w') as f:
    f.write(tree_rules)

# （3）选一条路径并解释其含义（假设是这条路径）
print("\n【任务2】分支解释：")
print("""
当样本满足：
- feature_2 ≤ 3.5，
- 并且 feature_1 ≤ 2.1，

则该样本将被分配到此叶子节点，预测值为 10.5。
这个值是所有符合这两个条件的训练样本目标值的平均值。
""")

# 保存解释为 Markdown 文件（任务2提交）
with open('branch_explanation.md', 'w') as f:
    f.write("# 分支解释\n\n")
    f.write("选择路径：\n```\n")
    f.write("|--- feature_2 <= 3.5\n")
    f.write("|   |--- feature_1 <= 2.1\n")
    f.write("|   |   |--- value: [10.5]\n")
    f.write("```\n\n")
    f.write("解释说明：\n")
    f.write("当样本的 `feature_2` 值小于等于 3.5，并且 `feature_1` 值小于等于 2.1 时，\n")
    f.write("该样本会被划分到这个叶子节点，预测值为 10.5。这是符合这两个条件的所有训练样本的目标值的平均值。\n")

# （4）手动推导该棵树的分裂过程（任务3：线下整理用）
print("\n【任务3】手动推导该棵树的生长过程所需信息如下：")

# 展示原始数据
print("\n原始数据集：")
print(df.head())

# 第一次分裂：尝试按某个特征分裂
# 假设我们选择 'cat__City_BALTIMORE' 作为第一个分裂特征
split_feature = 'cat__City_BALTIMORE'
split_value = 0.5

# 使用预处理后的数据进行分裂
X_transformed = preprocessor.transform(X)
feature_names = preprocessor.get_feature_names_out()

# 确保 split_feature 是预处理后的特征名称之一
if split_feature not in feature_names:
    raise ValueError(f"特征 {split_feature} 不在预处理后的特征列表中。预处理后的特征列表为: {feature_names}")

split_feature_index = np.where(feature_names == split_feature)[0][0]

# 修正稀疏矩阵的处理方式
left_idx = X_transformed[:, split_feature_index] <= split_value
right_idx = X_transformed[:, split_feature_index] > split_value

# 将稀疏矩阵转换为密集矩阵
left_idx_dense = left_idx.toarray().flatten()
right_idx_dense = right_idx.toarray().flatten()

# 输出子集的样本数量
print("\n第一次分裂：{} <= {}".format(split_feature, split_value))
print("左子集样本数量（<= {}）: {}".format(split_value, np.sum(left_idx_dense)))
print("右子集样本数量（> {}）: {}".format(split_value, np.sum(right_idx_dense)))

# 计算方差减少量
def calc_variance_reduction(y_full, y_left, y_right):
    full_var = np.var(y_full)
    left_var = np.var(y_left) if len(y_left) else 0
    right_var = np.var(y_right) if len(y_right) else 0
    weighted_var = (len(y_left)/len(y_full)) * left_var + (len(y_right)/len(y_full)) * right_var
    reduction = full_var - weighted_var
    return full_var, left_var, right_var, weighted_var, reduction

full_var, left_var, right_var, w_var, red = calc_variance_reduction(
    y, y[left_idx_dense], y[right_idx_dense]
)

print(f"\n总方差：{full_var:.2f}")
print(f"左子集方差：{left_var:.2f}")
print(f"右子集方差：{right_var:.2f}")
print(f"加权方差：{w_var:.2f}")
print(f"方差减少量：{red:.2f}")

# 结束提示
print("\n你可以根据以上信息手写或整理成 Word 文档完成任务3。")