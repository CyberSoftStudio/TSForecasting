import numpy as np
import matplotlib.pyplot as plt
import pywt


def derivative(signal):
    return signal[1:] - signal[:-1]


def derivativen(signal, n):
    result = signal + 0
    for i in range(n):
        result = derivative(result)

    return result


def extrapolation(arr, new_size):
    interval = np.linspace(0, new_size - 1, len(arr) + 1)
    result = [0] * new_size
    for i in range(len(arr)):
        for j in range(int(interval[i]), int(interval[i + 1])):
            try:
                result[j] = arr[i]
            except:
                pass

    print(len(result))
    return result


def mult_rows(arr, num):
    return [arr] * num


with open('tmp.txt', 'r') as f:
    line = f.readline()
    price_series = [float(x) for x in line[1:-2].split(',')]
    # print(line)
    n = len(price_series)
    # print(price_series)

# price_series = np.array(price_series)
price_series = np.array(price_series)[n - 256:n]


def resample(decomposition, window_size = 1024):
    # if window_size == None:
    #     window_size =
    resampled_decomposition = []
    scale = int(window_size / len(decomposition))

    koef = 1
    i = 3
    for x in decomposition[1:]:
        # koef = 1
        resampled_decomposition.extend(mult_rows(koef * np.array(extrapolation(x, window_size)), scale))
        koef += i
        i += 2

    resampled_decomposition = np.array(resampled_decomposition)
    return resampled_decomposition


def wavelet_filtration(window, mod, wname='db6'):
    decomposition = pywt.wavedec(window, wname)
    for i in range(min(len(mod), len(decomposition))):
        if mod[i] == 0:
            decomposition[i].fill(0)

    return pywt.waverec(decomposition, wname)

plt.figure()
plt.plot(price_series)
price_series = wavelet_filtration(price_series, [1], wname='db3')

plt.plot(price_series, 'r')

l = 0
r = 1
m = 1

extremums = []

while r < len(price_series):
    if price_series[r] == price_series[m]:
        r += 1
    elif price_series[r] > price_series[m] > price_series[l]:
        m += 1
        l += 1
    elif price_series[r] < price_series[m] < price_series[l]:
        m += 1
        l += 1
    elif price_series[r] < price_series[m] > price_series[l]:
        extremums.append((l, r, 'max'))
        l = r - 1
        m = r
        r = r + 1
    elif price_series[r] > price_series[m] < price_series[l]:
        extremums.append((l, r, 'min'))
        l = r - 1
        m = r
        r = r + 1

print(extremums)

plt.show()