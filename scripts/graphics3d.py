import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pywt

plt.style.use('dark_background')

with open('tmp.txt', 'r') as f:
    line = f.readline()
    price_series = [float(x) for x in line[1:-2].split(',')]

    # print(line)
    n = len(price_series)
    # print(price_series)

images = []

window_size = 256

b = 2560

# plt.figure()

window = price_series[n - window_size - b: n - b]
window = np.array(window)

# plt.plot(window)

# fig = plt.figure()
# ax = Axes3D(fig)

decomposition, _ = pywt.cwt(window, np.arange(1,256), 'gaus6')

# print(decomposition)
# decomposition = np.array(decomposition)

print(decomposition.shape)

x=np.arange(1, 257)
y=np.arange(1, 256)

X, Y = np.meshgrid(x, y)

a = [()]

print(X.shape, Y.shape, decomposition.shape)
tmp =np.abs(decomposition)
# tmp = decomposition
# ax.plot_surface(X, Y, np.abs(decomposition), cmap='plasma', vmax=abs(tmp).max(), vmin=abs(tmp).min())
# plt.matshow(np.log(decomposition))

cmaps = [('Perceptually Uniform Sequential', [
            'viridis', 'plasma', 'inferno', 'magma']),
         ('Sequential', [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
         ('Sequential (2)', [
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper']),
         ('Diverging', [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
         ('Qualitative', [
            'Pastel1', 'Pastel2', 'Paired', 'Accent',
            'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c']),
         ('Miscellaneous', [
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'])]

cmap_type, cmap_list = cmaps[0]
# for cmap_name in cmap_list:
#     fig = plt.figure()
#     ax = Axes3D(fig)
#     ax.plot_surface(X, Y, np.abs(decomposition), cmap=cmap_name, vmax=tmp.max(), vmin=tmp.min())
#     ax.set_title(cmap_type + ' ' + cmap_name)

#1 viridis
#2 coppper hot afmhot bone
#3 diverging spectral
#4 tab20b
#5 ocean cubehelix terrain gist_earth


comaps = ['viridis', 'copper', 'hot', 'afmhot', 'bone', 'Spectral', 'tab20b', 'ocean', 'cubehelix', 'terrain', 'gist_earth']
for cmap_name in comaps:
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, np.abs(decomposition), cmap=cmap_name, vmax=tmp.max(), vmin=tmp.min())
    ax.set_title(cmap_name)


#all : terrain, gist_earth, Spectral, afmhot

final_choice = ['terrain', 'gist_earth', 'Spectral', 'afmhot']
#
# for cmap_name in final_choice:
#     fig = plt.figure()
#     ax =  Axes3D(fig)
#     # # ax = fig.gca('3d', axesbg = 'gray')
#     # ax = fig.add_subplot(111, projection='3d')
#     # plt.gca().patch.set_facecolor('white')
#     # ax.w_xaxis.set_pane_color((1, 1, 1, 0.5))
#     # ax.w_yaxis.set_pane_color((0, 0, 0, .4))
#     # ax.w_zaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
#     ax.plot_surface(X, Y, np.abs(decomposition), cmap=cmap_name, vmax=tmp.max(), vmin=tmp.min())
#     ax.set_title(cmap_name)
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.plot_surface(X, Y, np.angle(decomposition))
plt.show()

