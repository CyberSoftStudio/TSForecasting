import numpy as np
import pywt
import math


def extrapolation(arr, new_size):
    interval = np.linspace(0, new_size - 1, len(arr) + 1)
    result = [0] * new_size
    for i in range(len(arr)):
        for j in range(int(interval[i]), int(interval[i + 1])):
            try:
                result[j] = arr[i]
            except:
                pass

    return result


def mult_rows(arr, num):
    return [arr] * num


with open('tmp.txt', 'r') as f:
    line = f.readline()
    price_series = [float(x) for x in line[1:-2].split(',')]
    # print(line)
    n = len(price_series)
    # print(price_series)

images = []

print(n)
window_size = 256
windows = []
koef = 1
i = 3

num_test = 16
c1, c2, c3 = 0,0,0
with open("prepared_data_3c.txt", 'w') as data, open("labels_3c.txt", "w") as labels:
    for b in range(n - window_size - num_test):
        windows.append(price_series[b: b + window_size])

        # print(';'.join(str(list(price_series[b + window_size: b + window_size + 16])).split(', '))[1:-1], file=labels)

        # if price_series[b + window_size] - price_series[b + window_size - 1] > 0:
        #     print(1, file=labels)
        # else:
        #     print(0, file=labels)
        eps = 0.001
        tmp1 = price_series[b + window_size] - price_series[b + window_size - 1]
        tmp2 = (price_series[b + window_size] + price_series[b + window_size - 1]) / 2
        print(tmp1, tmp2, eps * tmp2)
        if tmp1 > eps * tmp2:
            print("1;0;0", file=labels)
            c1 += 1
        elif math.fabs(tmp1) < eps * tmp2:
            print("0;1;0", file=labels)
            c2 += 1
        else:
            print("0;0;1", file=labels)
            c3 += 1

        decomposition = pywt.wavedec(windows[-1], 'db6')

        resampled_decomposition = []

        for x in decomposition[1:-1]:
            resampled_decomposition.extend(extrapolation(x, window_size))
            koef += i
            i += 2

        print(';'.join(str(list(price_series[b: b + window_size])).split(', '))[1:-1], file=data)

print(c1, c2, c3)

