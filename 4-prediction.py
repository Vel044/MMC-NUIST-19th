import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

# 读取数据
df = pd.read_csv('A题数据\merged_data_with_validation.csv')
df['年月'] = pd.to_datetime(df['年月'])

# 数据划分
df_train = df[df['年月'] < '2023-01-01']
df_valid = df[(df['年月'] >= '2023-01-01') & (df['年月'] <= '2023-12-01')]
df_future = df[df['年月'] > '2023-12-01']
cities = df['城市'].unique()

# 保存误差结果
mse_table = []

for city in cities:
    # 获取训练和验证数据
    train = df_train[df_train['城市'] == city][['年月', '空气质量综合指数']].copy()
    valid = df_valid[df_valid['城市'] == city][['年月', '空气质量综合指数']].copy()
    future = df_future[df_future['城市'] == city][['年月', '空气质量综合指数']].copy()

    # Prophet 预测
    prophet_df = train.rename(columns={'年月': 'ds', '空气质量综合指数': 'y'})
    model = Prophet(yearly_seasonality=True)
    model.fit(prophet_df)
    future_df = model.make_future_dataframe(periods=24, freq='MS')  # 一次性生成2023和2024共24个月

    forecast = model.predict(future_df)
    forecast_valid = forecast[['ds', 'yhat']].set_index('ds').loc[valid['年月']].reset_index()
    mse_prophet = mean_squared_error(valid['空气质量综合指数'].values, forecast_valid['yhat'].values)

    # SARIMA 预测
    sarima_model = sm.tsa.SARIMAX(train['空气质量综合指数'], order=(1,1,1), seasonal_order=(1,1,1,12),
                                  enforce_stationarity=False, enforce_invertibility=False)
    sarima_result = sarima_model.fit(disp=False)
    sarima_forecast = sarima_result.get_forecast(steps=12)
    sarima_pred = sarima_forecast.predicted_mean
    mse_sarima = mean_squared_error(valid['空气质量综合指数'].values, sarima_pred)

    # 保存误差
    mse_table.append({
        '城市': city,
        'Prophet_MSE': mse_prophet,
        'SARIMA_MSE': mse_sarima,
        '更优模型': 'Prophet' if mse_prophet < mse_sarima else 'SARIMA'
    })
    plt.rc('font', family='SimHei')

    # 画对比图
    plt.figure(figsize=(10, 6))
    plt.plot(valid['年月'], valid['空气质量综合指数'], label='实际值', marker='o')
    plt.plot(valid['年月'], forecast_valid['yhat'], label='Prophet预测', linestyle='--', marker='x')
    plt.plot(valid['年月'], sarima_pred.values, label='SARIMA预测', linestyle='-.', marker='s')
    plt.title(f'{city} 空气质量指数预测对比 (2023)')
    plt.xlabel('时间')
    plt.ylabel('空气质量指数')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{city}_comparison_2023.png')
    plt.close()

    # 用更优模型预测 2024 年
    if mse_prophet < mse_sarima:
        # Prophet预测未来
        forecast_2024 = forecast[['ds', 'yhat']].set_index('ds').loc['2024-01-01':'2024-12-01']
        pred_2024 = forecast_2024['yhat'].values
    else:
        sarima_forecast_2024 = sarima_result.get_forecast(steps=24)
        pred_2024 = sarima_forecast_2024.predicted_mean[-12:].values

    # 画2024年预测图
    months_2024 = pd.date_range(start='2024-01-01', periods=12, freq='MS')
    plt.figure(figsize=(8, 5))
    plt.plot(months_2024, pred_2024, marker='o')
    plt.title(f'{city} 2024年空气质量预测 ({ "Prophet" if mse_prophet < mse_sarima else "SARIMA" })')
    plt.xlabel('时间')
    plt.ylabel('预测空气质量指数')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{city}_forecast_2024.png')
    plt.close()

# 保存误差表格
mse_df = pd.DataFrame(mse_table)
mse_df.to_csv('模型MSE比较结果.csv', index=False)
print("所有预测已完成，图像和误差表格已保存。")

# ✅ 对比所有城市 Prophet 与 SARIMA 的 MSE
plt.figure(figsize=(12, 6))
bar_width = 0.35
index = np.arange(len(mse_df))

# 设置中文显示
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 画柱状图
plt.bar(index, mse_df['SARIMA_MSE'], width=bar_width, label='SARIMA MSE', color='skyblue')
plt.bar(index + bar_width, mse_df['Prophet_MSE'], width=bar_width, label='Prophet MSE', color='salmon')

# 设置图形属性
plt.xlabel('城市')
plt.ylabel('MSE（均方误差）')
plt.title('SARIMA 与 Prophet 模型在各城市的 MSE 对比')
plt.xticks(index + bar_width / 2, mse_df['城市'], rotation=45)
plt.legend()
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('总体MSE对比图.png')
plt.close()

print("✅ 总体 MSE 对比图已保存为 '总体MSE对比图.png'")
