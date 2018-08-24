import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import pywt
from scripts.differentiationlib import get_extremums2d, wavedec_filtration, mult_rows, extrapolation
import scipy.signal as signal
from scripts.extremumlib import *
from scripts.differentiationlib import wavedec_filtration


def get_cwt(window, mask = (0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0), wdname = 'db6', wcname = 'morl', scale = 40, wl = -1, wr = -1):
    window = wavedec_filtration(window, mask, wdname)
    decomposition, _ = pywt.cwt(window, np.arange(1, scale), wcname)

    tmp = np.abs(decomposition)
    phi = np.cos(np.angle(decomposition))

    return tmp * phi


plt.style.use('dark_background')

with open('tmp.txt', 'r') as f:
    line = f.readline()
    price_series = [float(x) for x in line[1:-2].split(',')]
    n = len(price_series)



window_size = 256
scale = 40

x = np.arange(1, window_size + 1)
y = np.arange(1, scale)

X, Y = np.meshgrid(x, y)

# plt.figure()

for b in range(0, 2560, 256):
    window = price_series[n - window_size - b: n - b]
    window = np.array(window)

    plt.figure()
    plt.plot(window)
    # with open("../images/windows/window{}.png".format(b), 'w') as file:
    #     pass
    # plt.savefig("../images/windows/window{}.png".format(b))
    tmp = get_cwt(window, scale=scale, mask=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    M = create_map(window, scale=scale, wl=-1, wr=-1, mask=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    M = linear(M)
    M = bound_filter(M, alpha=0.25)
    M = signal.medfilt2d(M, (1, 3))

    intervals = get_patterns2d(M)

    img = np.zeros(M.shape)

    for i in range(len(intervals)):
        for x in intervals[i]:
            img[i, x[0]:x[1] + 1] = M[i, x[0]:x[1] + 1]

    plt.matshow(img)
    # with open('../images/patterns_1l_m1_m1/pattern{}.png'.format(b), 'w') as file:
    #     pass
    # plt.savefig('../images/patterns_1l_m1_m1/pattern{}.png'.format(b))
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, tmp, cmap='ocean')
    with open("../images/cwt_1l/cwt{}.png".format(b), 'w') as file:
        pass
    plt.savefig("../images/cwt_1l/cwt{}.png".format(b))

# plt.show()