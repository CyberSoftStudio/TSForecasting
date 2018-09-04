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

jsonfile = '../../KerasCNN/models/model_cwt.json' #model_nclasses_46_1
h5file   = '../../KerasCNN/models/model_cwt.h5'

# cnn = load_cnn(jsonfile, h5file)
model_json_path = '../../KerasCNN/models/model_cwt_10k.json' #model_nclasses_46_1
model_h5_path   = '../../KerasCNN/models/model_cwt_10k.h5'
json_file = open(model_json_path, 'r')
loaded_model_json = json_file.read()
json_file.close()
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


with open('../train_data/temp.txt', 'r') as f:
    lines = list(f)
    price_series = [float(x) for x in lines[0][1:-2].split(',')]
    price_times = [str(x) for x in lines[1][1:-2].split(",")]
    # print(line)
    n = len(price_series)
    # print(price_series)

price_series = signal = np.sin(np.linspace(0, 20 * math.pi, 256))


def predict(window, scale=50, assurance=0.9, wdname='db6', wcname='morl'):
    print("i am in predict function")
    block_sizex = 32
    block_sizey = 32

    assert len(window) >= block_sizey + 10
    try:
        window = np.array(window)
    except:
        return [],[]

    M = get_cwt_swt(window,
                    scale=scale,
                    mask=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    wdname=wdname,
                    wcname=wcname
                    )

    M = linear(M)
    # M = bound_filter(M, alpha=0.5)

    test = []
    coords = []

    for i in range(scale - block_sizex):
        for j in range(0,M.shape[1]-block_sizey + 1):
            print(i,i + block_sizex, j,j + block_sizey)
            test.append(M[i:i + block_sizex, j: j + block_sizey])
            coords.append((i, j))

    test = np.array(test)
    print(test)
    test = test.reshape(test.shape[0], block_sizex, block_sizey, 1)
    result = cnn.predict(test, verbose=1)

    cnt = 0
    wow = []
    wow_coords = []
    for i in range(len(result)):
        if result[i, 2] > assurance:
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
        segmentations[-1].extract(alpha=0.5)

    return correct_rects, segmentations, M


correct_rects, segmentations, M = predict(price_series[-256:], assurance=0.5, wdname='dmey')
fig, ax = plt.subplots()
plt.matshow(M, fignum=False)

for i in range(len(correct_rects)):
    cur_rect = correct_rects[i]
    cur_coords = (cur_rect.x[1], cur_rect.x[0])

    r = patches.Rectangle(cur_coords, cur_rect.w, cur_rect.h, edgecolor='red', facecolor='none', alpha=0.8)
    ax.add_patch(r)


plt.show()