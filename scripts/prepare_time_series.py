import pandas as pd
import numpy as np

data = pd.read_csv('last_btc_usd_100000.csv', sep=',')

data_time = data['time'].values
data_prices = data['price'].values
data_volume = data['time'].values

price_series = []

n = 5256
dt = (data_time[-1] - data_time[0]) / n
c = 0

for i in range(len(data_time)):
    t = data_time[i]
    p = data_prices[i]
    if c < int((t - data_time[0]) / dt):
        price_series.append(p)
        c += 1


# for t, p in zip(data_time, data_prices):
#     if c < int((t - data_time[0]) / dt):
#         price_series.append(p)
#         c += 1

price_series.append(data_prices[-1])
price_series = np.array(price_series)

print(list(price_series), file=open('tmp1.txt', 'w'))