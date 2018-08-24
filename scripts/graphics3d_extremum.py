import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pywt
from scripts.differentiationlib import get_extremums2d

plt.style.use('dark_background')

with open('tmp.txt', 'r') as f:
    line = f.readline()
    price_series = [float(x) for x in line[1:-2].split(',')]

    # print(line)
    n = len(price_series)
    # print(price_series)

images = []

window_size = 256

b = 256 * 4

scale = 40

# plt.figure()

window = price_series[n - window_size - b: n - b]
window = np.array(window)

plt.figure()
plt.plot(window - 400)

decomposition_dwt = pywt.wavedec(window, 'db6')
decomposition_dwt[0].fill(0)
decomposition_dwt[-1].fill(0)
decomposition_dwt[-2].fill(0)
decomposition_dwt[-3].fill(0)
# decomposition_dwt[-4].fill(0)
# decomposition_dwt[-5].fill(0)

window = pywt.waverec(decomposition_dwt, 'db6')

plt.plot(window)

decomposition, _ = pywt.cwt(window, np.arange(1, scale), 'morl')

x=np.arange(1, window_size + 1)
y=np.arange(1, scale)

X, Y = np.meshgrid(x, y)

tmp = np.abs(decomposition)

phi = np.cos(np.angle(decomposition))
# plt.matshow(phi)
# fig = plt.figure()
# ax= Axes3D(fig)
# ax.scatter(X, Y, phi)

tmp = tmp * phi

# print(tmp.shape)
#
extremums = get_extremums2d(tmp)

print(len(extremums))
#
# extremums_max = [[x[2] for x in extremum_list if x[3] == 'max'] for extremum_list in extremums]
# extremums_min = [[x[2] for x in extremum_list if x[3] == 'min'] for extremum_list in extremums]
#
# print(extremums_max)
# print(extremums_min)
#
# plt.figure()
# plt.plot(np.array(extremums_max[0]))
# plt.plot(np.array(extremums_min[0]))
# plt.figure()
# plt.plot(np.array(tmp[:, 0]))
print(extremums)

new_tmp = np.ones(tmp.shape)
l = 0
r = 0
for i in range(tmp.shape[1]):
    for extremum in extremums[i]:
        if True:

            new_tmp[max(0, extremum[0] - l): min(new_tmp.shape[0], extremum[1] + 1 + r), i] = tmp[max(0, extremum[0] - l): min(new_tmp.shape[0], extremum[1] + 1 + r), i]
            # m = extremum[0] + extremum[1]
            # m = m // 2
            # new_tmp[m, i] = tmp[m, i]

plt.matshow(new_tmp)

sumbytime = new_tmp.sum(axis=0)

plt.figure()
plt.plot(sumbytime)

tmp = new_tmp
phi_max = tmp.max(axis=1)
time_max = tmp.max(axis=0)
plt.figure()
plt.subplot(211)
plt.plot(time_max)
time_max = tmp.min(axis=0)
plt.plot(time_max)
plt.subplot(212)
plt.plot(phi_max)
phi_max = tmp.min(axis=1)
plt.plot(phi_max)

final_choice = ['terrain','ocean']
#
for cmap_name in final_choice:
    fig = plt.figure()
    ax =  Axes3D(fig)
    # # ax = fig.gca('3d', axesbg = 'gray')
    # ax = fig.add_subplot(111, projection='3d')
    # plt.gca().patch.set_facecolor('white')
    # ax.w_xaxis.set_pane_color((1, 1, 1, 0.5))
    # ax.w_yaxis.set_pane_color((0, 0, 0, .4))
    # ax.w_zaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
    ax.plot_surface(X, Y, tmp, cmap=cmap_name, vmax=tmp.max(), vmin=tmp.min())
    ax.set_title(cmap_name)
    # ax.scatter(X, Y, tmp, color='r')
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.plot_surface(X, Y, np.angle(decomposition))
plt.show()

