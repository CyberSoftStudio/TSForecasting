"""
i have many samples and i have a cnn that can detect patterns in images
so i can prepare much bigger dataset from some source and run cnn through
every square that is not pattern i can add to bad class
evere square without continuation by other square is also in bad class
all other could be recommended to good class
question: how to find out that square without continuation?
answer:
After merging all rects, squares which sizes are 32x32 will stand alone they are in bad class

Now we try to create concept for needed functions
we need to solve several problems
we must prepare new dataset
we must run through it with cnn
we must prepare three classes which are bad class without patterns,
bad class with pattern but without continuation and
good class with both pattern and continuation

we now how to marge rects, we have complete class for it
we now how to check if square without continuation

All of that gives as an algorithm

1) prepare bigger dataset (probably, several millions samples)
2) decompose it by frames
3) for each frame find it cwt and run cnn on it
4) find all rects and merge them if it is needed
5) each square that is not a pattern we can add to a first class
6) each rect that is only 32x32 we can add to a second class
7) each rect that is not 32x32 we can add to a good class

"""


import random
import numpy as np
import pandas as pd

# Keras Imports - CNN
from keras.models import model_from_json

# My scripts import
from scripts.extremumlib import *
from scripts.Segmentation import *
from scripts.Rect import *

import time
import datetime
import json


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

with open('temp.txt', 'r') as f:
    lines = list(f)
    price_series = [float(x) for x in lines[0][1:-2].split(',')]
    # print(line)
    n = len(price_series)
    # print(price_series)

images = []

window_size = 256

intervals = []

bad_first_class = []
bad_second_class = []
good_class = []

for b in range(0, len(price_series), 256):

    scale = 50

    x = np.arange(1, window_size + 1)
    y = np.arange(1, scale)

    X, Y = np.meshgrid(x, y)

    window = price_series[n - window_size - b: n - b]
    window = np.array(window)

    M = get_cwt_swt(window, scale=scale, mask=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    M = linear(M)
    # M = bound_filter(M, alpha=0.5)

    block_sizex = 32
    block_sizey = 32

    test = []
    coords = []

    for i in range(scale - block_sizex):
        for j in range(window_size - block_sizey):
            test.append(M[i:i + block_sizex, j:j + block_sizey])
            coords.append((i, j))

    test = np.array(test)
    test = test.reshape(test.shape[0], block_sizex, block_sizey, 1)
    result = cnn.predict(test, verbose=1)

    cnt = 0
    wow = []
    wow_coords = []

    for i in range(len(result)):
        if result[i, 1] > 0.80:
            cnt+=1
            wow.append(test[i, :, :, 0])
            wow_coords.append(coords[i])
        else:
            bad_first_class.append(test[i, :, :, 0])

    segmentations = []

    wow_rects = [Rect(wow_coords[i], 32, 32) for i in range(cnt)]

    wow_rects = sorted(wow_rects, key=lambda a: a.x[1])

    correct_rects = []

    for rect in wow_rects:
        if len(correct_rects) == 0:
            correct_rects.append(rect)
        elif not correct_rects[-1].is_crossing(rect):
            correct_rects.append(rect)
        else:
            correct_rects[-1] = correct_rects[-1].get_convex_rect(rect)

    for x in correct_rects:
        if x.h == x.w == 32:
            bad_second_class.append(M[x.x[0]: x.x[0] + x.h, x.x[1]: x.x[1] + x.w])
        else:
            good_class.append(M[x.x[0]: x.x[0] + 32, x.x[1]: x.x[1] + 32])

    wow = [M[x.x[0]: x.x[0] + x.h, x.x[1]: x.x[1] + x.w] for x in correct_rects]


data = []
for x in bad_first_class[:1000]:
    data.append((x, 0))

for x in bad_second_class:
    data.append((x, 1))

for x in good_class:
    data.append((x, 2))


def prepare_data(data):
    new_data = []
    for x, y in data:
        x = np.array(list(x.reshape((1, -1))[0, :]) + [y])
        # print(len(x))
        new_data.append(x)
    return pd.DataFrame(data=new_data)

# size = len(data)
# for i in range(size):
#     data.append((-data[i][0], data[i][1]))

print(len(data), len(bad_first_class), len(bad_second_class), len(good_class))
data = prepare_data(data)
data.to_csv('../../KerasCNN/input/data_cwt.csv', sep=',', header=None, index=None)



