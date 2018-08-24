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

import time
import datetime
import json


def show_segment(plane: np.array, paths: [np.array], x = 0, y = 0) -> None:

    for path in paths:
        print(path)
        plt.plot(path[:, 1] + x, path[:, 0] + y)


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
    price_times = [str(x) for x in lines[1][1:-2].split(",")]
    # print(line)
    n = len(price_series)
    # print(price_series)

images = []

window_size = 256

intervals = []

for b in range(0, 10 * 256, 256):

    # b = 0  # 1792

    scale = 40

    x = np.arange(1, window_size + 1)
    y = np.arange(1, scale)

    X, Y = np.meshgrid(x, y)

    # plt.figure()

    window = price_series[n - window_size - b: n - b]
    window = np.array(window)

    tmptimes = price_times[n - window_size - b: n - b]
    tmptimes = np.array(tmptimes)
    # print(tmptimes)

    ptimes = []
    for i in range(len(tmptimes)):

        # try:
        ptimes.append(time.mktime(datetime.datetime.strptime(str(tmptimes[i]), " '%Y.%m.%d %H:%M:%S'").timetuple()))
        # except:
            # print(tmptimes[i])
    print(ptimes)
    #
    # plt.figure()
    # plt.plot(window)
    M = get_cwt(window, scale=scale, mask=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    # plt.matshow(M)
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

    print(cnt)

    fig, ax = plt.subplots()
    plt.matshow(M, fignum=False)
    # plt.plot(window)

    segmentations = []

    for i in range(cnt):
        cur_coords = (wow_coords[i][1], wow_coords[i][0])

        segmentations.append(Segmentation(wow[i]))
        segmentations[-1].extract(alpha=0.5)

        show_segment(*segmentations[-1].show(), x=cur_coords[0], y=cur_coords[1])

        cur_intervals = segmentations[-1].get_intervals()
        segmentations[-1].recalc_separators()
        cur_intervals = list(filter(lambda x: x[1] - x[0] > 5, cur_intervals))

        # for x in cur_intervals:
        #     if x[-1] == 1.0:
        #         plt.axvline(x=x[0] + cur_coords[0], color='g')
        #         plt.axvline(x=x[1] + cur_coords[0], color='g')
        #     else:
        #         plt.axvline(x=x[0] + cur_coords[0], color='b')
        #         plt.axvline(x=x[1] + cur_coords[0], color='b')

        xcoords = segmentations[-1].separators + cur_coords[0]
        print(xcoords)

        begin_time = ptimes[0]
        dt = 300
        for i in range(len(cur_intervals)):
            cur_intervals[i] = (cur_intervals[i][0] * dt + begin_time, cur_intervals[i][1] * dt + begin_time, cur_intervals[i][2])

        # print(cur_intervals)
        intervals.append(cur_intervals)

        r = patches.Rectangle(cur_coords, 32, 32, edgecolor='red', facecolor='none', alpha=0.8)
        ax.add_patch(r)

# print(intervals, file = open("intervals.json", 'w'))
result = {"intervals":intervals}
json.dump(result, open("intervals.json", 'w'))
    # for x in segmentations:
    #     x.extract(alpha=0.5)
    #     # show_segment(*x.show())




plt.show()
