import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
validation_data = pd.read_csv('A题数据/validation.csv')
merged_data = pd.read_csv('A题数据/merged_data.csv')

# Convert '年月' (Year-Month) to datetime for proper sorting
merged_data['年月'] = pd.to_datetime(merged_data['年月'], format='%Y年%m月')
validation_data['年月'] = pd.to_datetime(validation_data['年月'], format='%Y年%m月')

# Sort merged_data by city and date
merged_data = merged_data.sort_values(by=['城市', '年月'])

# Split data into training (2018-2022) and test (2023) sets
train_data = merged_data[merged_data['年月'] < '2023-01-01']
test_data = merged_data[merged_data['年月'] >= '2023-01-01']

# Get unique cities
cities = merged_data['城市'].unique()

# Get initial air quality values for December 2022
initial_A = train_data[train_data['年月'] == '2022-12-01'].set_index('城市')['空气质量综合指数']

# Dictionary to store predictions and MSE
predictions = {}
mse_dict = {}

# Features for sensitivity analysis
features = ['产成品', '工业用电量', '常住人口', '月降水量', '绿化覆盖率']

# Model training and prediction for each city
for city in cities:
    # Prepare training data
    city_train_data = train_data[train_data['城市'] == city].copy()
    city_train_data['A_next'] = city_train_data['空气质量综合指数'].shift(-1)
    city_train_data = city_train_data.dropna()  # Drop last row with no next value
    
    # Define features and target
    X_train = city_train_data[['空气质量综合指数'] + features].astype(float)
    y_train = city_train_data['A_next'].astype(float)
    
    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Prepare test data for prediction
    city_test_data = test_data[test_data['城市'] == city].copy()
    city_predictions = []
    A_prev = initial_A[city]
    
    # Predict air quality for each month in 2023
    for index, row in city_test_data.iterrows():
        X_pred = pd.DataFrame({
            '空气质量综合指数': [A_prev],
            **{feat: [row[feat]] for feat in features}
        }).astype(float)
        A_pred = model.predict(X_pred)[0]
        city_predictions.append(A_pred)
        A_prev = A_pred
    
    predictions[city] = city_predictions
    
    # Calculate MSE
    actual = validation_data[city].values
    mse = np.mean((actual - city_predictions) ** 2)
    mse_dict[city] = mse
    print(f'{city} 的均方误差 (MSE): {mse:.4f}')

# Find the city with the best prediction (lowest MSE)
best_city = min(mse_dict, key=mse_dict.get)
print(f'\n预测效果最好的城市是 {best_city}，MSE = {mse_dict[best_city]:.4f}')

# Plot actual vs predicted values for the best city only
plt.rc('font', family='SimHei')
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.figure(figsize=(10, 6))
plt.plot(validation_data.index, validation_data[best_city], label='Actual', marker='o', color='blue')
plt.plot(validation_data.index, predictions[best_city], label='Predicted', marker='x', color='red')
plt.title(f'Air Quality Index Prediction for {best_city} (2023)')
plt.xlabel('Month')
plt.ylabel('Air Quality Comprehensive Index')
plt.legend()
plt.grid(True)
plt.savefig(f'{best_city}_air_quality_prediction.png')
plt.close()

# Sensitivity Analysis
sensitivity_results = {}

for city in cities:
    # Use the last month's data for sensitivity analysis
    last_row = test_data[test_data['城市'] == city].iloc[-1]
    A_prev = predictions[city][-1]  # Last predicted value
    
    # Original prediction
    X_original = pd.DataFrame({
        '空气质量综合指数': [A_prev],
        **{feat: [last_row[feat]] for feat in features}
    }).astype(float)
    original_pred = model.predict(X_original)[0]
    
    sensitivity = {}
    for feat in features:
        # Positive perturbation (+1%)
        X_pos = X_original.copy()
        X_pos[feat] *= 1.01
        pred_pos = model.predict(X_pos)[0]
        
        # Negative perturbation (-1%)
        X_neg = X_original.copy()
        X_neg[feat] *= 0.99
        pred_neg = model.predict(X_neg)[0]
        
        # Calculate sensitivity as the average relative change
        sensitivity[feat] = np.mean([abs(pred_pos - original_pred), abs(pred_neg - original_pred)]) / original_pred
    
    sensitivity_results[city] = sensitivity

# Create a DataFrame for sensitivity results
sensitivity_df = pd.DataFrame(sensitivity_results).T
sensitivity_df = sensitivity_df[features]  # Reorder columns

# Display sensitivity table
print('\n敏感性分析结果（相对变化率）：')
print(sensitivity_df)

# Plot sensitivity bar chart for the best city

plt.figure(figsize=(10, 6))
sns.barplot(data=sensitivity_df.loc[best_city].sort_values(ascending=False), palette='coolwarm')
plt.title(f'Sensitivity Analysis for {best_city}')
plt.xlabel('Features')
plt.ylabel('Relative Change in Prediction')
plt.grid(True)
plt.savefig(f'{best_city}_sensitivity_analysis.png')
plt.close()

# Create a DataFrame for MSE
mse_df = pd.DataFrame(list(mse_dict.items()), columns=['城市', 'MSE'])
mse_df = mse_df.sort_values(by='MSE')



# Display the MSE table
print('\n模型指标表格（按MSE升序排列）：')
print(mse_df)



# Plot MSE bar chart using Seaborn
plt.figure(figsize=(12, 6))
sns.barplot(x='城市', y='MSE', data=mse_df, palette='viridis')
plt.title('Mean Squared Error (MSE) for Each City')
plt.xlabel('City')
plt.ylabel('MSE')
plt.xticks(rotation=45)
plt.grid(True)
plt.savefig('mse_bar_chart.png')
plt.close()

print("\nPrediction, validation, and sensitivity analysis completed. Plots saved as PNG files.")

# 将 validation_data 中的数据补充回 merged_data 的 2023 年部分
merged_data_with_validation = merged_data.copy()

# 遍历每个城市，把 validation_data 中的 2023 年数据填入 merged_data 中
for i, month in enumerate(pd.date_range('2023-01-01', '2023-12-01', freq='MS')):
    for city in cities:
        condition = (merged_data_with_validation['城市'] == city) & (merged_data_with_validation['年月'] == month)
        merged_data_with_validation.loc[condition, '空气质量综合指数'] = validation_data.loc[i, city]

# 保存到新文件
merged_data_with_validation.to_csv('A题数据/merged_data_with_validation.csv', index=False, encoding='utf-8-sig')

print("\n已将 validation 数据填入 merged_data 并保存为 merged_data_with_validation.csv。")


# 计算每个城市2018-2023年的平均空气质量指数
avg_aqi = merged_data.groupby('城市')['空气质量综合指数'].mean().sort_values()

# 可视化：平均空气质量直方图
plt.figure(figsize=(12, 6))
sns.barplot(x=avg_aqi.index, y=avg_aqi.values, palette='coolwarm')

plt.title('2018–2023年各城市平均空气质量综合指数')
plt.xlabel('城市')
plt.ylabel('平均空气质量综合指数')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('average_air_quality_bar.png')
plt.close()

print("\n已生成各城市平均空气质量直方图：average_air_quality_bar.png")



from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# 保存标准化回归系数与随机森林重要性
coef_table = pd.DataFrame()
rf_importance_table = pd.DataFrame()

for city in cities:
    city_train_data = train_data[train_data['城市'] == city].copy()
    city_train_data['A_next'] = city_train_data['空气质量综合指数'].shift(-1)
    city_train_data = city_train_data.dropna()
    
    X = city_train_data[['空气质量综合指数'] + features].astype(float)
    y = city_train_data['A_next'].astype(float)

    # 标准化处理
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 标准化回归模型
    std_model = LinearRegression()
    std_model.fit(X_scaled, y)
    coefs = std_model.coef_
    
    coef_df = pd.DataFrame({
        '城市': city,
        '变量': ['空气质量综合指数'] + features,
        '标准化系数': coefs
    })
    coef_table = pd.concat([coef_table, coef_df], ignore_index=True)

    # 随机森林重要性分析
    rf = RandomForestRegressor(random_state=42, n_estimators=100)
    rf.fit(X, y)
    importances = rf.feature_importances_

    rf_df = pd.DataFrame({
        '城市': city,
        '变量': ['空气质量综合指数'] + features,
        '重要性': importances
    })
    rf_importance_table = pd.concat([rf_importance_table, rf_df], ignore_index=True)

# 保存表格
coef_table.to_csv('A题数据/standardized_coefficients.csv', index=False, encoding='utf-8-sig')
rf_importance_table.to_csv('A题数据/random_forest_importance.csv', index=False, encoding='utf-8-sig')

# 可视化：每个城市的标准化回归系数图
for city in cities:
    city_coef = coef_table[coef_table['城市'] == city].sort_values(by='标准化系数', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='标准化系数', y='变量', data=city_coef, palette='coolwarm')
    plt.title(f'{city} - 标准化回归系数（越大越重要）')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{city}_standardized_coefficients.png')
    plt.close()

# 可视化：每个城市的随机森林重要性图
for city in cities:
    city_rf = rf_importance_table[rf_importance_table['城市'] == city].sort_values(by='重要性', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='重要性', y='变量', data=city_rf, palette='viridis')
    plt.title(f'{city} - 随机森林特征重要性')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{city}_random_forest_importance.png')
    plt.close()

print("\n已完成标准化回归系数与随机森林重要性分析，并生成每个城市的条形图。")
