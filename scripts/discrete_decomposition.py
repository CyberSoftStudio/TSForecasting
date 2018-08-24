import numpy as np
import pywt
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import map_coordinates
import math

with open('tmp.txt', 'r') as f:
    line = f.readline()
    price_series = [float(x) for x in line[1:-2].split(',')]
    # print(line)
    n = len(price_series)
    # print(price_series)

images = []

window_size = 3000

b = 500
window = price_series[n - window_size - b: n - b]

plt.figure()
plt.plot(window, color = 'r')

decomposition = pywt.wavedec(window, 'db6')


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


resampled_decomposition = []
scale = int(window_size / len(decomposition))

koef = 1
i = 3
for x in decomposition[1:]:
    koef = 1
    resampled_decomposition.extend(mult_rows(koef * np.array(extrapolation(x, window_size)), scale))
    koef += i
    i += 2

resampled_decomposition = np.array(resampled_decomposition)

plt.figure()
plt.plot(pywt.waverec(decomposition, 'db6'))


for i in range(16):
    new_decomposition = [np.array(decomposition[0], copy=True)]
    for j in range(1, len(decomposition)):
        new_decomposition.append(np.array(decomposition[j], copy=True))

        if i & int(1 ** j - 1) == 0:
            new_decomposition[-1].fill(0)

    # plt.figure(i + 3)
    # plt.plot(pywt.waverec(new_decomposition, 'db6'))


plt.figure()
# plt.imshow(resampled_decomposition, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',vmax=abs(resampled_decomposition).max(), vmin=-abs(resampled_decomposition).max())
plt.imshow(resampled_decomposition)
plt.colorbar()

plt.show()
