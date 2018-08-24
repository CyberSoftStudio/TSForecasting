import random
import numpy as np
import pandas as pd

# Visualisation imports
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Keras Imports - CNN
from keras.models import model_from_json

# My scripts import
from scripts.extremumlib import *
from scripts.Segmentation import *


def show_segment(plane: np.array, paths: [np.array]) -> None:
    plt.figure()
    plt.matshow(plane, fignum=False)
    for path in paths:
        print(path)
        plt.plot(path[:, 1], path[:, 0])


# load json and create

model_json_path = '../../KerasCNN/models/model1.json' #model_nclasses_46_1
model_h5_path   = '../../KerasCNN/models/model1.h5'
json_file = open(model_json_path, 'r')
loaded_model_json = json_file.read()
json_file.close()
cnn = model_from_json(loaded_model_json)
# load weights into new model
cnn.load_weights(model_h5_path)
print("Loaded model from disk")


opt = 'adam'
loss = 'categorical_crossentropy'
metrics = ['accuracy']
# Compile the classifier using the configuration we want
cnn.compile(optimizer=opt, loss=loss, metrics=metrics)


plt.style.use('dark_background')

with open('temp.txt', 'r') as f:
    lines = list(f)
    price_series = [float(x) for x in lines[0][1:-2].split(',')]
    price_times = [x for x in lines[1][1:-2].split(',')]
    # print(line)
    n = len(price_series)
    # print(price_series)

images = []

window_size = 256

b = 0  # 1792

scale = 40

x = np.arange(1, window_size + 1)
y = np.arange(1, scale)

X, Y = np.meshgrid(x, y)

# plt.figure()

window = price_series[n - window_size - b: n - b]
window = np.array(window)

times = price_series[n - window_size - b: n - b]
times = np.array(times)

#
plt.figure()
plt.plot(window)
M = get_cwt(window, scale=scale, mask=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
plt.matshow(M)
# M = create_map(window, scale=scale, wl=0, wr=0, mask = [0, 1, 1, 0, 0, 0, 0, 0, 0, 0])
M = linear(M)
# M = bound_filter(M, alpha=0.5)

block_sizex = 32
block_sizey = 32

test = []
times = []
coords = []
for i in range(scale - block_sizex):
    for j in range(window_size - block_sizey):
        test.append(M[i:i + block_sizex, j:j + block_sizey])
        times.append(j)
        coords.append((i, j))
test = np.array(test)
test = test.reshape(test.shape[0], block_sizex, block_sizey, 1)
result = cnn.predict(test, verbose=1)

# print(result[0])
# plt.figure()
# plt.plot(result[0])

cnt = 0
wow = []
nah = []
wow_times = []
wow_coords = []
for i in range(len(result)):
    if result[i, 1] > 0.80:
        cnt+=1
        wow.append(test[i, :, :, 0])
        wow_times.append(times[i])
        wow_coords.append(coords[i])
    else:
        nah.append(test[i, :, :, 0])

print(cnt)

fig, ax = plt.subplots()
plt.matshow(M, fignum=False)

segmentations = []

for i in range(cnt):
    cur_coords = (wow_coords[i][1], wow_coords[i][0])
    segmentations.append(Segmentation(wow[i]))
    print(cur_coords)
    r = patches.Rectangle(cur_coords, 32, 32, edgecolor='red', facecolor='none', alpha=0.8)
    ax.add_patch(r)

for x in segmentations:
    x.extract(alpha=0.5)
    show_segment(*x.show())


# for k in range(len(wow) // 10):
#     fig = plt.figure()
#     for i in range(10):
#         ax = fig.add_subplot(2, 5, i + 1)
#         plt.matshow(wow[k * 10 + i], fignum=False)
#     fig = plt.figure()
#     for i in range(10):
#         ax = fig.add_subplot(2, 5, i + 1)
#         plt.plot(window[wow_times[k * 10 + i]:wow_times[k * 10 + i] + block_sizey])
#
# fig = None
# if len(wow) % 10:
#     fig = plt.figure()
# for k in range(len(wow) % 10):
#     ax =fig.add_subplot(1, len(wow) % 10, k + 1)
#     plt.matshow(wow[len(wow) - len(wow) % 10 + k], fignum=False)

# intervals = get_patterns2d(M)

# img = np.zeros(M.shape)

# for i in range(len(intervals)):
#     for x in intervals[i]:
#         img[i, x[0]:x[1] + 1] = M[i, x[0]:x[1] + 1]
#
# plt.matshow(img)
# print(len(intervals), intervals)
# # plt.figure()
# # plt.plot(window)
#
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.plot_surface(X, Y, M, cmap='terrain')

plt.show()
