import numpy as np
# My scripts import
from scripts.Segmentation import Segmentation
from scripts.Rect import Rect
from scripts.extremumlib import get_cwt, get_cwt_swt, linear, bound_filter

from keras.models import model_from_json
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D

from scripts.prediction_lib import predict, predict_interval, load_cnn


model_json_path = '../../KerasCNN/models/model_cwt_10k.json' #model_nclasses_46_1
model_h5_path   = '../../KerasCNN/models/model_cwt_10k.h5'

load_cnn(model_json_path, model_h5_path)

with open('../train_data/eurusd_m5.txt', 'r') as f:
    lines = list(f)
    price_series = [float(x) for x in lines[0][1:-2].split(',')]
    price_times = [str(x) for x in lines[1][1:-2].split(",")]
    # print(line)
    n = len(price_series)
    # print(price_series)

#
mult_const = 4
window_size = 256 * mult_const
shift = 30 * mult_const
scale = 50 * mult_const
wdname='db6'
wcname='gaus8'
extract_alpha = 0.5

for b in range(shift + 1,50 * 30 + 1, 10):
    correct_rects, segmentations, M = predict(
                                            price_series[-window_size - b: -b],
                                            scale=scale,
                                            assurance=0.5,
                                            wdname=wdname,
                                            wcname=wcname,
                                            shift=shift,
                                            extract_alpha=extract_alpha,
                                            key=1,
                                            mult_const=mult_const
                                        )

    fig = plt.figure()
    ax = fig.add_subplot(311)
    plt.matshow(M, fignum=False)

    for i in range(len(correct_rects)):
        cur_rect = correct_rects[i]
        cur_coords = (cur_rect.x[1], cur_rect.x[0])

        r = patches.Rectangle(cur_coords, cur_rect.w, cur_rect.h, edgecolor='red', facecolor='none', alpha=0.8)
        ax.add_patch(r)

    True_M = get_cwt_swt(price_series[-window_size - b + shift: -b + shift],
                    scale=scale,
                    mask=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    wdname=wdname,
                    wcname=wcname
                    )
    True_M = linear(True_M)
    ax = fig.add_subplot(312)
    plt.matshow(True_M, fignum=False)
    for i in range(len(correct_rects)):
        cur_rect = correct_rects[i]

        cur_rect.x *= mult_const
        cur_rect.w *= mult_const
        cur_rect.h *= mult_const

        cur_coords = (cur_rect.x[1], cur_rect.x[0])

        r = patches.Rectangle(cur_coords, cur_rect.w, cur_rect.h, edgecolor='red', facecolor='none', alpha=0.8)
        ax.add_patch(r)

    if len(segmentations):
        try:
            interval = predict_interval(segmentations[0].segmentation)
            x, y = correct_rects[0].x

            interval['maxy'] *= mult_const
            interval['miny'] *= mult_const
            interval['center'] *= mult_const

            interval['maxy'] += y
            interval['miny'] += y
            interval['center'] += y

            plt.axvline(x=interval['miny'])
            plt.axvline(x=interval['maxy'])
        except:
            print("Can't predict interval")

    ax = fig.add_subplot(313)
    plt.plot(price_series[-window_size - b + shift: -b + shift])

    if len(segmentations):
        try:
            interval = predict_interval(segmentations[0].segmentation)
            x, y = correct_rects[0].x

            interval['maxy'] *= mult_const
            interval['miny'] *= mult_const
            interval['center'] *= mult_const

            interval['maxy'] += y
            interval['miny'] += y
            interval['center'] += y

            plt.axvline(x=interval['miny'])
            plt.axvline(x=interval['maxy'])
        except:
            print("Can't predict interval")


    plt.show()
