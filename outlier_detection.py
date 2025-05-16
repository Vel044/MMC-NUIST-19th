import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv('A题数据/附件1：江苏省各市2018年7月-2022年12月空气质量综合指数.csv', encoding='gbk')

# 转换日期格式
df['日期'] = pd.to_datetime(df['日期'].str.replace('.', '-') + '-01')

# 异常值检测函数
def detect_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5*iqr
    upper_bound = q3 + 1.5*iqr
    return (series < lower_bound) | (series > upper_bound)

# 对各城市数据进行检测
cities = df.columns[1:]
outliers = pd.DataFrame()

for city in cities:
    # 转换为数值型（处理特殊字符）
    df[city] = pd.to_numeric(df[city].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
    city_outliers = df[detect_outliers(df[city])]
    if not city_outliers.empty:
        outliers = pd.concat([outliers, city_outliers])

# 输出异常值
print('发现异常值记录：')
print(outliers.drop_duplicates().sort_values('日期'))

# 可视化（可选）
import matplotlib.pyplot as plt
for city in cities:
    plt.figure(figsize=(12,4))
    plt.plot(df['日期'], df[city], label='正常值')
    plt.scatter(outliers['日期'], outliers[city], color='red', label='异常值')
    plt.title(f'{city}空气质量综合指数异常检测')
    plt.legend()
    plt.show()