import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取 MSE 比较结果文件
mse_df = pd.read_csv('模型MSE比较结果.csv')

# 设置中文字体（根据你的环境调整）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

cities = mse_df['城市'].values
prophet_mse = mse_df['Prophet_MSE'].values
sarima_mse = mse_df['SARIMA_MSE'].values

x = np.arange(len(cities))  # x轴刻度位置
width = 0.35  # 条形宽度

fig, ax = plt.subplots(figsize=(12, 6))
bar1 = ax.bar(x - width/2, prophet_mse, width, label='Prophet MSE')
bar2 = ax.bar(x + width/2, sarima_mse, width, label='SARIMA MSE')

ax.set_xlabel('城市')
ax.set_ylabel('均方误差 (MSE)')
ax.set_title('各城市 Prophet 与 SARIMA 模型 MSE 对比')
ax.set_xticks(x)
ax.set_xticklabels(cities, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

# 在条形顶部显示具体数值
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 偏移量
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

autolabel(bar1)
autolabel(bar2)

plt.tight_layout()
plt.show()
