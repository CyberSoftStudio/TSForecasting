import numpy as np
import pywt
from scripts.differentiationlib import get_extremums2d, wavedec_filtration, mult_rows, extrapolation, swt_filtration


def detrend(window, wname='db6'):
    return wavedec_filtration(window, [0], wname=wname)


def get_trend(window, wname='db6'):
    return wavedec_filtration(window, [1] + [0 for i in range(20)], wname=wname)


def resample(A, size):
    res = []
    old_size = A.shape
    print(size, old_size, size[0]/old_size[0])
    for i in range(old_size[0]):
        res.extend(mult_rows(extrapolation(A[i, :], size[1]), round(size[0] / old_size[0])))
    print(res)
    return np.array(res)


def fftfilter(image, core):
    newcore = np.zeros(image.shape)
    newcore[:core.shape[0], :core.shape[1]] = core
    newimage = np.fft.fft2(image)
    newcore = np.fft.fft2(newcore)
    filtimage = newimage * newcore
    result = np.fft.ifft2(filtimage)
    return result.real


def bound_filter(A, alpha=0.25):
    M = A.copy()
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            # print(M[i][j])
            if abs(M[i, j]) <= alpha:
                # print("i am here")
                M[i, j] = 0
    return M


def sigmoid(x):
    return np.exp(x) / (np.exp(x) + 1)


def tanh(x):
    return np.tanh(x)


def linear(X):
    return X / np.abs(X).max()


def linear_normal(X):
    return linear(X - X.min())


def get_cwt(window, mask = (0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0), wdname='db6', wcname='morl', scale=40):
    window = wavedec_filtration(window, mask, wdname)
    decomposition, _ = pywt.cwt(window, np.arange(1, scale), wcname)

    tmp = np.abs(decomposition)
    phi = np.cos(np.angle(decomposition))

    return tmp * phi


def get_cwt_swt(window, mask = (0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0), wdname='db6', wcname='morl', scale=40):
    window = detrend(window, wdname)
    window = swt_filtration(window, mask, wdname)
    decomposition, _ = pywt.cwt(window, np.arange(1, scale), wcname)

    tmp = np.abs(decomposition)
    phi = np.cos(np.angle(decomposition))

    return tmp * phi


def create_map(window, mask = (0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0), wdname = 'db6', wcname = 'morl', scale = 40, wl = -1, wr = -1):
    window = wavedec_filtration(window, mask, wdname)
    decomposition, _ = pywt.cwt(window, np.arange(1, scale), wcname)

    tmp = np.abs(decomposition)
    phi = np.cos(np.angle(decomposition))

    tmp = tmp * phi

    extremums = get_extremums2d(tmp)
    new_tmp = np.zeros(tmp.shape)

    for i in range(tmp.shape[1]):
        for extremum in extremums[i]:
            new_tmp[max(0, extremum[0] - wl): min(new_tmp.shape[0], extremum[1] + 1 + wr), i] = tmp[max(0, extremum[0] - wl): min(new_tmp.shape[0], extremum[1] + 1 + wr), i]

    return new_tmp


def satisfy_pattern(pattern):
    return abs((pattern[np.array([0, 2, 4])] - np.ones(3) * pattern[0])).max() < 2 and abs((pattern[np.array([1,3])] - np.ones(2) * pattern[1])).max() < 2


def normalize_pattern_intervals(intervals):
    def crossing(a, b):
        return a[1] > b[0]

    def unite(a, b):
        return (a[0], b[1])

    res = []
    for x in intervals:
        if len(res) == 0:
            res.append(x)
        elif crossing(res[-1], x):
            res[-1] = unite(res[-1], x)
        else:
            res.append(x)

    return res


def get_patterns(input):
    x = np.sign(input).astype(np.int8) + 1

    size = len(x)
    intervals = []
    prep = []
    l = 0
    r = 1
    cnt = [0, 0, 0]
    cnt[x[0]] += 1

    for i in range(1, size):
        if x[i] != x[i - 1]:
            for j in range(len(cnt)):
                if cnt[j] > 0:
                    prep.append((j - 1, cnt[j]))
                    cnt[j] = 0
        cnt[x[i]] += 1

    for i in range(len(cnt)):
        if cnt[i] > 0:
            prep.append((i - 1, cnt[i]))
            cnt[i] = 0

    # print(prep)

    """
    Теперь мы можем пройтись окном по пять так мы получим +0-0+ при этом у паттернов эти вещи будут примерно одной длины
    """
    prep = np.array(prep)
    steps = [0] + list(prep[:, 1].copy())
    for i in range(1, len(steps)):
        steps[i] += steps[i - 1]
    for i in range(4, prep.shape[0]):
        mask = prep[i - 4: i + 1, 0]
        value = prep[i - 4: i + 1, 1]
        print(value, satisfy_pattern(value))

        if satisfy_pattern(value): #and (not (mask == np.array([1, 0, -1, 0, 1])).any() or not (mask == np.array([-1, 0, 1, 0, -1])).any()):
            intervals.append((steps[i - 4], steps[i + 1]))

    return intervals


def get_patterns2d(M):
    intervals = []

    for i in range(M.shape[0]):
        intervals.append(normalize_pattern_intervals(get_patterns(M[i,:])))

    return intervals