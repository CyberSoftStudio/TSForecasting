import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt


def gen_sample():
    sample = np.zeros((20,32))

    for i in range(sample.shape[0]):
        for j in range(sample.shape[1]):
            tmp = random.uniform()
            if tmp < 0.2:
                sample[i, j] = -1
            elif tmp < 0.4:
                sample[i, j] = 1
            else:
                sample[i, j] = 0

    return sample


def can_put(A, block, pos):
    try:
        tmp = A[pos[0]:pos[0] + block.shape[0], pos[1]:pos[1] + block.shape[1]]
    except:
        return False

    return np.abs(tmp).max() == 0

blocks = [
    [[1, 1]],
    [[0, 1, 1],
     [1, 1, 0]],
    [[1,1,1],
     [1,1,1]]

]

for i in range(10):
    plt.matshow(gen_sample())

plt.show()