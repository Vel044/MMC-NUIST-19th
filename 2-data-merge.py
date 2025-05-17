import pandas as pd
from pathlib import Path

def clean_city(city_series):
    return (city_series.str.replace(r'[a-zA-Z]+', '', regex=True)
            .str.replace(r'市$|市区$|\s+', '', regex=True)
            .str.strip())

# 处理常住人口（修复列名清洗问题）
pop = pd.read_csv(Path('ExternalData/按地区分常住人口.csv'), encoding='utf-8-sig')
pop = pop.melt('年份', var_name='原始城市', value_name='常住人口')
pop['城市'] = clean_city(pop['原始城市'])  # 加强清洗逻辑
pop_expanded = []
for _, row in pop.iterrows():
    for m in range(1, 13):
        pop_expanded.append({
            '城市': row['城市'],
            '年月': f"{row['年份']}-{m:02d}",
            '常住人口': row['常住人口']
        })
pop_df = pd.DataFrame(pop_expanded)

# 处理工业用电量
power = pd.read_csv(Path('ExternalData/工业用电量.CSV'), encoding='utf-8-sig')
power = power.melt('地区Region', var_name='年份', value_name='工业用电量')
power['城市'] = clean_city(power['地区Region'])
power['年份'] = power['年份'].astype(int)
power_expanded = []
for _, row in power.iterrows():
    for m in range(1, 13):
        power_expanded.append({
            '城市': row['城市'],
            '年月': f"{row['年份']}-{m:02d}",
            '工业用电量': row['工业用电量']
        })
power_df = pd.DataFrame(power_expanded)

# 处理工业产成品
prod = pd.read_csv(Path('ExternalData/工业产成品.csv'), encoding='utf-8-sig')
prod['城市'] = clean_city(prod['年份地区YearCity'].str.split('市').str[0])
prod = prod[['城市', '产成品FinishedGoods']].rename(columns={'产成品FinishedGoods': '产成品'})

# 处理绿化
green = pd.read_csv(Path('ExternalData/绿化情况.csv'), encoding='utf-8-sig')
green['城市'] = clean_city(green['年份Year城市City'])
green = green[['城市', '人均公园绿地面积', '建成区绿化覆盖率']].rename(columns={'建成区绿化覆盖率': '绿化覆盖率'})

# 处理用水
water = pd.read_csv(Path('ExternalData/用水量.csv'), encoding='utf-8-sig')
water['城市'] = clean_city(water['年份Year城市City'])
water = water[['城市', '生产用水量', '人均日生活用水量']]

# 处理降水
precip = pd.read_csv(Path('ExternalData/月降水量.CSV'), encoding='utf-8-sig')
precip = precip.melt('城 市City', var_name='月份', value_name='月降水量')
precip['城市'] = clean_city(precip['城 市City'])
precip['月份'] = precip['月份'].str.extract(r'(\d+)').astype(int)
precip_expanded = []
for _, row in precip.iterrows():
    for y in range(2018, 2024):  # 扩展到2023年
        precip_expanded.append({
            '城市': row['城市'],
            '年月': f"{y}-{row['月份']:02d}",
            '月降水量': row['月降水量']
        })
precip_df = pd.DataFrame(precip_expanded)

# 处理气温
temp = pd.read_csv(Path('ExternalData/月平均气温.CSV'), encoding='utf-8-sig')
temp = temp.melt('城 市City', var_name='月份', value_name='月平均气温')
temp['城市'] = clean_city(temp['城 市City'])
temp['月份'] = temp['月份'].str.extract(r'(\d+)').astype(int)
temp_expanded = []
for _, row in temp.iterrows():
    for y in range(2018, 2024):  # 扩展到2023年
        temp_expanded.append({
            '城市': row['城市'],
            '年月': f"{y}-{row['月份']:02d}",
            '月平均气温': row['月平均气温']
        })
temp_df = pd.DataFrame(temp_expanded)

# 处理空气质量
air = pd.read_csv(Path('插值结果/插值_ARIMA.csv'), parse_dates=['日期'])
air['年月'] = air['日期'].dt.strftime('%Y-%m')
air = air.melt(id_vars=['年月'], value_vars=air.columns[1:-1], 
              var_name='城市', value_name='空气质量综合指数')
air['城市'] = clean_city(air['城市'])

# 为每个城市添加2023年的12个月数据，空气质量综合指数为空
cities = air['城市'].unique()
for city in cities:
    for m in range(1, 13):
        air = air._append({
            '年月': f"2023-{m:02d}",
            '城市': city,
            '空气质量综合指数': None
        }, ignore_index=True)

# 合并数据
merged = air.merge(pop_df, on=['城市', '年月'], how='left').merge(power_df, on=['城市', '年月'], how='left')
merged = merged.merge(prod, on='城市', how='left').merge(green, on='城市', how='left')
merged = merged.merge(water, on='城市', how='left').merge(precip_df, on=['城市', '年月'], how='left')
merged = merged.merge(temp_df, on=['城市', '年月'], how='left')

# 调整列顺序
cols = ['城市', '年月', '产成品', '工业用电量', '人均公园绿地面积', '绿化覆盖率',
        '生产用水量', '人均日生活用水量', '月降水量', '月平均气温', '常住人口', '空气质量综合指数']
merged = merged[cols]

# 按城市和年月排序
merged = merged.sort_values(by=['城市', '年月'])

# 合并后删除原日期行
merged = merged[merged['城市'] != '原日期']

merged.to_csv('ExternalData/合并数据.csv', index=False, encoding='utf-8-sig')
print('数据合并完成，保存为 合并数据.csv')