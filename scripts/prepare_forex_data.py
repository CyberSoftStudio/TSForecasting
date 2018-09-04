import pandas as pd
import numpy as np
from datetime import datetime
import time

data = pd.read_csv('../exchange_data/EURUSD_M5_2.csv', sep='\t')

print(data.head())

closes = data['<CLOSE>'].values

assert isinstance((data['<DATE>'] + ' ' + data['<TIME>']).values, object)
times = (data['<DATE>'] + ' ' + data['<TIME>']).values

with open("../train_data/eurusd_m5_2.txt", 'w')as file:
    print(str(list(closes)), file=file)
    print(str(list(times)), file=file)