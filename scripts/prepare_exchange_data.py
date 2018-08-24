import pandas as pd
import numpy as np


def prepare_exchange_data(input_file, save_file, sep=',', header=None, rec_num=100000):
    exchange_data = pd.read_csv(input_file, sep=sep, header=header)
    data = exchange_data.loc[-rec_num:, :]
    data.columns = ['time', 'price', 'value']
    data.to_csv(save_file, sep=sep, index=None)


exchanges = [
 'btc_usd_datasets/bitfinexUSD.csv'
]

last_records_files = [
    'last_records/last_100000_btc_usd_bitfinex.csv'
]

prepare_exchange_data("btc_usd_datasets/bitfinexUSD.csv", 'last_btc_usd_100000.csv')


