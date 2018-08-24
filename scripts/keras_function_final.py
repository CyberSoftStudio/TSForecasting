import numpy as np
# My scripts import
from scripts.Segmentation import Segmentation
from scripts.Rect import Rect
from scripts.extremumlib import get_cwt_swt, linear

from keras.models import model_from_json

jsonfile = '../../KerasCNN/models/model_cwt.json' #model_nclasses_46_1
h5file   = '../../KerasCNN/models/model_cwt.h5'

# cnn = load_cnn(jsonfile, h5file)
model_json_path = '../../KerasCNN/models/model_cwt.json' #model_nclasses_46_1
model_h5_path   = '../../KerasCNN/models/model_cwt.h5'
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


def predict(window, scale=50):
    print("i am in predict function")
    block_sizex = 32
    block_sizey = 32

    assert len(window) >= block_sizey
    try:
        window = np.array(window)
    except:
        return [],[]

    M = get_cwt_swt(window, scale=scale, mask=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    M = linear(M)
    # M = bound_filter(M, alpha=0.5)

    test = []
    coords = []

    for i in range(scale - block_sizex):
        for j in range(-block_sizey,-block_sizey + 1):
            print(i,i + block_sizex, j,j + block_sizey)
            test.append(M[i:i + block_sizex, M.shape[1] + j: M.shape[1] + j + block_sizey])
            coords.append((i, j))

    test = np.array(test)
    print(test)
    test = test.reshape(test.shape[0], block_sizex, block_sizey, 1)
    result = cnn.predict(test, verbose=1)

    cnt = 0
    wow = []
    wow_coords = []
    for i in range(len(result)):
        if result[i, 2] > 0.90:
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

    return correct_rects, segmentations
