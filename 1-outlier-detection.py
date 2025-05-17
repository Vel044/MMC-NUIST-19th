import pandas as pd
import matplotlib.pyplot as plt

# 1. 读入，把日期列当字符串处理
df = pd.read_csv(
    'A题数据/附件1：江苏省各市2018年7月-2022年12月空气质量综合指数.csv',
    encoding='utf-8',
    dtype={'日期': str}
)
cities = df.columns.drop('日期')

# 2. 清洗：去掉城市数据里的非数字/小数点字符，转 float
df[cities] = (
    df[cities]
      .astype(str)
      .replace(r'[^0-9\.]', '', regex=True)
      .apply(pd.to_numeric, errors='coerce')
)

# 3. 严格抽出"YYYY.MM"（两位月）
#    这样连 "\,"、" " 这些都一并丢掉，保证只剩下形如 "2018.10" 的文本
df['原日期'] = df['日期'].str.extract(r'(\d{4}\.\d{2})')[0]

# 4. 解析成时间戳（默认当月第一天）
df['日期'] = pd.to_datetime(df['原日期'], format='%Y.%m', errors='coerce')
df.dropna(subset=['日期'], inplace=True)

# 5. 异常值检测（IQR×3倍距）
def detect_outliers(s: pd.Series, k: float = 3.0) -> pd.Series:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (s < q1 - k * iqr) | (s > q3 + k * iqr)

# 6. 收集 & 打印所有城市的异常点
outs = []
for city in cities:
    m = detect_outliers(df[city])
    if m.any():
        tmp = df.loc[m, ['日期', city]].copy()
        tmp['城市'] = city
        outs.append(tmp)
if outs:
    out_df = pd.concat(outs, ignore_index=True).sort_values(['城市','日期'])
    print("\n=== 异常值记录（IQR×3） ===")
    print(out_df.to_markdown(index=False))
else:
    print("未检测到任何异常值。")

# 7. 可视化：将所有城市的折线图绘制在一张图上
plt.rc('font', family='SimHei')

# 创建一个大图
plt.figure(figsize=(15, 8))

# 绘制所有城市的折线图
for city in cities:
    plt.plot(df['日期'], df[city], label=city)
    
    # 标记异常值
    m = detect_outliers(df[city])
    if m.any():
        plt.scatter(df.loc[m, '日期'], df.loc[m, city], s=50, marker='x')

# 设置图表标题和标签
plt.title('江苏省各市空气质量综合指数时间序列')
plt.xlabel('日期')
plt.ylabel('空气质量综合指数')
plt.grid(True, linestyle='--', alpha=0.7)

# 添加图例，并将其放在图表外部右侧
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 自动调整布局
plt.tight_layout()

# 显示图表
plt.show()

# 8. 可选：单独绘制每个城市的异常值图（保留原来的单独图表功能）
'''
for city in cities:
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(df['日期'], df[city], label='值')
    m = detect_outliers(df[city])
    if m.any():
        ax.scatter(df.loc[m,'日期'], df.loc[m,city], s=50, c='red', label='异常')
    ax.set(title=f'{city} 空气质量综合指数', xlabel='日期', ylabel='指数')
    ax.legend()
    plt.tight_layout()
    plt.show()
'''


from scipy.interpolate import CubicSpline
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import os

# 缺失值统一设为 NaN，包括异常值
for city in cities:
    m = detect_outliers(df[city])
    df.loc[m, city] = np.nan

# 保存四种插值结果
df_cubic = df.copy()
df_arima = df.copy()
df_idw = df.copy()
df_kriging = df.copy()

# 三次样条插值
for city in cities:
    valid = df['日期'][df[city].notna()]
    values = df[city][df[city].notna()]
    cs = CubicSpline(valid.map(pd.Timestamp.toordinal), values)
    df_cubic[city] = df['日期'].map(pd.Timestamp.toordinal).map(cs)

# ARIMA插值
for city in cities:
    s = df[city].copy()
    for i in range(len(s)):
        if pd.isna(s.iloc[i]):
            train = s.iloc[:i].dropna()
            if len(train) < 10:
                continue  # 太短就跳过
            try:
                model = ARIMA(train, order=(1, 1, 0)).fit()
                pred = model.forecast(steps=1)
                s.iloc[i] = pred.values[0]
            except:
                continue
    df_arima[city] = s

# IDW 插值
def idw_interpolate(series, timestamps, power=2):
    x_known = timestamps[series.notna()]
    y_known = series[series.notna()]
    x_missing = timestamps[series.isna()]
    result = series.copy()
    for t in x_missing:
        dists = np.abs(x_known - t)
        weights = 1 / (dists ** power + 1e-6)
        result.loc[timestamps == t] = np.sum(weights * y_known) / np.sum(weights)
    return result

ordinals = df['日期'].map(pd.Timestamp.toordinal)

for city in cities:
    df_idw[city] = idw_interpolate(df[city], ordinals)




# Kriging 插值

# 加载城市坐标（你已提供 csv 文件 ExternalData/jiangsu_cities_coordinates.csv）
coord_df = pd.read_csv("ExternalData/jiangsu_cities_coordinates.csv")
city_to_coord = {row['城市']: (row['经度'], row['纬度']) for _, row in coord_df.iterrows()}

from pykrige.ok import OrdinaryKriging

from pykrige.ok import OrdinaryKriging

df_kriging = df.copy()

# 按时间横向做 Kriging（空间维度）
for i, row in df.iterrows():
    # 当前这一行对应一个时间点
    values = []
    lons = []
    lats = []
    for city in cities:
        val = row[city]
        if not np.isnan(val):
            coord = city_to_coord.get(city)
            if coord is not None:
                lon, lat = coord
                lons.append(lon)
                lats.append(lat)
                values.append(val)

    # 至少要5个有效城市才能做空间插值
    if len(values) < 5:
        continue

    try:
        OK = OrdinaryKriging(
            lons, lats, values,
            variogram_model='linear',
            verbose=False, enable_plotting=False
        )

        # 插值所有城市（包括缺失）
        interp_row = {}
        for city in cities:
            if pd.isna(row[city]):
                coord = city_to_coord.get(city)
                if coord is not None:
                    lon, lat = coord
                    z, _ = OK.execute('points', [lon], [lat])
                    interp_row[city] = z[0]

        # 更新到 df_kriging 这一行
        for city, val in interp_row.items():
            df_kriging.at[i, city] = val

    except Exception as e:
        print(f"{row['日期']} Kriging 插值失败: {e}")



# 写入文件
os.makedirs("插值结果", exist_ok=True)
df_cubic.to_csv("插值结果/插值_三次样条.csv", index=False)
df_arima.to_csv("插值结果/插值_ARIMA.csv", index=False)
df_idw.to_csv("插值结果/插值_IDW.csv", index=False)
df_kriging.to_csv("插值结果/插值_Kriging.csv", index=False)


# 可视化Kriging插值结果
plt.figure(figsize=(15, 8))

# 绘制原始数据和Kriging插值结果对比
for city in cities:
    # 绘制原始数据（半透明）
    plt.plot(df['日期'], df[city], '--', alpha=0.3, label=f'{city}原始')
    # 绘制Kriging插值结果
    plt.plot(df['日期'], df_kriging[city], '-', linewidth=2, label=f'{city}Kriging插值')

# 设置图表标题和标签
plt.title('江苏省各市空气质量综合指数Kriging插值结果')
plt.xlabel('日期')
plt.ylabel('空气质量综合指数')
plt.grid(True, linestyle='--', alpha=0.7)

# 添加图例（放在图表外部右侧）
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), 
          bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# 自动调整布局
plt.tight_layout()

# 保存图表
plt.savefig("插值结果/Kriging插值可视化.png", dpi=300, bbox_inches='tight')

# 显示图表
plt.show()