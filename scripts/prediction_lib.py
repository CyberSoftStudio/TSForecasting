import numpy as np
# My scripts import
from scripts.Segmentation import Segmentation, Segment
from scripts.Rect import Rect
from scripts.extremumlib import get_cwt, get_cwt_swt, linear, bound_filter, linear_normal

from keras.models import model_from_json
from PIL import Image
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import scipy.misc as misc
import scipy.ndimage as ndimage
import json
# cnn = load_cnn(jsonfile, h5file)
model_json_path = '../../KerasCNN/models/model_cwt_10k.json' #model_nclasses_46_1
model_h5_path   = '../../KerasCNN/models/model_cwt_10k.h5'

cnn = None


def load_cnn_flask(model_json_path, model_h5_path):
    global cnn

    loaded_model_json = None
    #with open(model_json_path) as json_file:
    #    loaded_model_json = json.dumps(json.load(json_file)))
    # json.dumps(json.load(open(model_json_path)))
    loaded_model_json = open(model_json_path).read()
    print(loaded_model_json)
    cnn = model_from_json(loaded_model_json)
    # load weights into new model
    cnn.load_weights(model_h5_path)
    cnn._make_predict_function()
    print("Loaded model from disk")

    opt = 'adam'
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']
    # Compile the classifier using the configuration we want
    cnn.compile(optimizer=opt, loss=loss, metrics=metrics)


def load_cnn(model_json_path, model_h5_path):
    global cnn
    json_file = open(model_json_path, 'rb')
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

    cnn._make_predict_function()


def predict(window, scale=50, assurance=0.9, wdname='db6', wcname='morl', shift = 30, block_sizex = 32, block_sizey = 32, key = 2, extract_alpha = 0.5, mult_const = 1, verbose=0):

    assert len(window) >= block_sizey + 10
    try:
        window = list(window)
        tmp = window[-shift:]
        tmp.reverse()
        window.extend(tmp)
        window = np.array(window[shift:])
    except Exception as e:
        print(e)
        return [],[],[]

    M = get_cwt_swt(window,
                    scale=scale,
                    mask=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    wdname=wdname,
                    wcname=wcname
                    )

    M = linear(M)
    M = ndimage.zoom(M, 1/mult_const)
    # M = misc.imresize(M, (49, 256))
    # M = linear(M)

    # print(M)
    shift = shift // mult_const
    scale = scale // mult_const


    # M.resize((50, 256))
    # M = bound_filter(M, alpha=0.5)

    test = []
    coords = []

    for i in range(scale - block_sizex):
        for j in range(M.shape[1] - block_sizey - shift, M.shape[1] - block_sizey - shift // 2 + 1):
            # print(i, i + block_sizex, j, j + block_sizey, M[i:i + block_sizex, j: j + block_sizey].shape)
            test.append(M[i:i + block_sizex, j: j + block_sizey])
            coords.append((i, j))

    test = np.array(test)
    # print(test.shape)
    test = test.reshape(test.shape[0], block_sizex, block_sizey, 1)
    result = cnn.predict(test, verbose=verbose)

    cnt = 0
    wow = []
    wow_coords = []
    for i in range(len(result)):
        if result[i, key] > assurance:
            cnt += 1
            wow.append(test[i, :, :, 0])
            wow_coords.append(coords[i])

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

    wow = [M[x.x[0]: x.x[0] + x.h, x.x[1]: x.x[1] + x.w] for x in correct_rects]
    segmentations = []

    for i in range(len(wow)):
        cur_rect = correct_rects[i]
        cur_coords = (cur_rect.x[1], cur_rect.x[0])

        segmentations.append(Segmentation(wow[i]))
        segmentations[-1].extract(alpha=extract_alpha)

    return correct_rects, segmentations, M


def predict_interval(segm):
    minimum = None
    maximum = None
    for s in segm:
        if s.type == 1.:
            try:
                if len(s.points) > len(maximum.points):
                    maximum = s
            except:
                maximum = s
        else:
            try:
                if len(s.points) > len(minimum.points):
                    minimum = s
            except:
                minimum = s

    maximum.recalc_convex_rect()
    minimum.recalc_convex_rect()

    assert maximum.maxy < minimum.miny

    predicted_interval = {
        'miny': minimum.maxy + minimum.miny - maximum.maxy,
        'maxy': minimum.maxy + minimum.miny - maximum.miny,
        'center': minimum.maxy + minimum.miny - (maximum.maxy + maximum.miny) // 2
    }

    return predicted_interval

