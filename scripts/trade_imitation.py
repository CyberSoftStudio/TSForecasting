import scripts.Bot as Bot
from scripts.extremumlib import get_trend
import numpy as np

intervals = []

with open('../train_data/btcusd_m5.txt', 'r') as f:
    lines = list(f)
    price_series = [float(x) for x in lines[0][1:-2].split(',')]
    # price_times = [str(x) for x in lines[1][1:-2].split(",")]
    # print(line)
    n = len(price_series)
    # print(price_series)

price_series = np.array(price_series)
window_size = 256

bot = Bot.Bot([])
bot.change_config({'assurance': 0.9})

limit = len(price_series) - window_size

for i in range(7000, limit):
    print('{}/{}'.format(i, limit))
    bot.set_data(price_series[i:i + window_size])

    interval = bot.predict()

    if len(interval) > 0:
        interval['miny'] += i
        interval['maxy'] += i
        interval['center'] += i

        print("Interval predicted", interval)
        intervals.append(interval)

balance = 1
cur_time = 0

print("Initial balance is", balance)

intervals = sorted(intervals, key=lambda x: (x['miny'], x['center'] - x['miny']))

trade_count = 0
average_balance = 0

bad_attempts = 0
bad_volume = 0

for x in intervals:
    if x['miny'] < cur_time:
        continue

    else:
        trade_count += 1
        #что на самом деле получается balance * (1/(price1 + alpha)) - balance * (price1 - alpha)
        fee = 0.000
        additional_balance = ((price_series[x['center']] - fee)/(price_series[x['miny']] + fee) - 1) * balance - 0.00015
        average_balance += abs(additional_balance)

        if additional_balance < 0:
            bad_attempts += 1
            bad_volume += abs(additional_balance)

        balance += additional_balance

        cur_time = x['center']

    print("Current balance is", balance)

print("End balance is", balance)

print("Statistic")
print("Trade count -", trade_count)
print("Average additional balance -", average_balance/trade_count)
print("Bad attempts count -", bad_attempts)
print("Negative changes -", bad_volume)