import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import pywt
from scripts.differentiationlib import get_extremums2d, wavedec_filtration, mult_rows, extrapolation
import scipy.signal as signal
from scripts.extremumlib import *


plt.style.use('dark_background')

with open('tmp.txt', 'r') as f:
    line = f.readline()
    price_series = [float(x) for x in line[1:-2].split(',')]

    # print(line)
    n = len(price_series)
    # print(price_series)

images = []

window_size = 256

b = 1792

scale = 80

x = np.arange(1, window_size + 1)
y = np.arange(1, scale)

X, Y = np.meshgrid(x, y)

# plt.figure()

window = price_series[n - window_size - b: n - b]
window = np.array(window)

plt.figure()
plt.plot(window)

M = create_map(window, scale=scale, wl=0, wr=0, mask = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
print(M)
M = linear(M)

plt.matshow(bound_filter(M))
num=2
core = np.array([[1 for i in range(num)] + [0] + [-1 for i in range(num)] + [0] + [1 for i in range(num)]])
core = resample(core, (6, 30))
# plt.matshow(core)


# M = fftfilter(M, core)
# M = linear(M)
M = bound_filter(M, alpha=0.25)
M = signal.medfilt2d(M, (1, 3))

intervals = get_patterns2d(M)

img = np.zeros(M.shape)

for i in range(len(intervals)):
    for x in intervals[i]:
        img[i, x[0]:x[1] + 1] = M[i, x[0]:x[1] + 1]

plt.matshow(img)
print(len(intervals), intervals)
# plt.figure()
# plt.plot(window)

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X, Y, M, cmap='terrain')


# plt.matshow(signal.medfilt2d(M, (1, 3)))
plt.show()
