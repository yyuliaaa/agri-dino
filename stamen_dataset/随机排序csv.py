import pandas as pd

# 读取Excel文件
df = pd.read_csv('train_stamen.csv')

# 随机排序行
df_shuffled = df.sample(frac=1).reset_index(drop=True)

# 保存到新的Excel文件
df_shuffled.to_csv('train_stamen.csv', index=False)