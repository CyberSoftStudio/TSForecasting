import pywt
import numpy as np


def derivative(signal):
    return signal[1:] - signal[:-1]


def derivativen(signal, n):
    result = signal + 0
    for i in range(n):
        result = derivative(result)

    return result


def extrapolation(arr, new_size):
    interval = np.linspace(0, new_size - 1, len(arr) + 1)
    result = [0] * new_size
    for i in range(len(arr)):
        for j in range(int(interval[i]), int(interval[i + 1])):
            try:
                result[j] = arr[i]
            except:
                pass

    print(len(result))
    return result


def mult_rows(arr, num):
    return [arr] * num


def resample(decomposition, window_size = 1024):
    # if window_size == None:
    #     window_size =
    resampled_decomposition = []
    scale = int(window_size / len(decomposition))

    koef = 1
    i = 3
    for x in decomposition[1:]:
        # koef = 1
        resampled_decomposition.extend(mult_rows(koef * np.array(extrapolation(x, window_size)), scale))
        koef += i
        i += 2

    resampled_decomposition = np.array(resampled_decomposition)
    return resampled_decomposition


def wavedec_filtration(window, mod, wname='db6'):
    decomposition = pywt.wavedec(window, wname)
    for i in range(min(len(mod), len(decomposition))):
        if mod[i] == 0:
            decomposition[i].fill(0)
    return pywt.waverec(decomposition, wname)


def swt_filtration(window, mod, wname='db6'):
    decomposition = pywt.swt(window, wname)
    for i in range(min(len(mod), len(decomposition))):
        if mod[i] == 0:
            decomposition[i][0].fill(0)
            decomposition[i][1].fill(0)
    return pywt.iswt(decomposition, wname)


def get_extremums(series):

    l = 0
    r = 1
    m = 1
    
    extremums = []

    while r < len(series):
        if series[r] == series[m]:
            r += 1
        elif series[r] > series[m] > series[l]:
            m += 1
            l += 1
        elif series[r] < series[m] < series[l]:
            m += 1
            l += 1
        elif series[r] < series[m] > series[l]:
            extremums.append((l, r, series[m], 'max'))
            l = r - 1
            m = r
            r = r + 1
        elif series[r] > series[m] < series[l]:
            extremums.append((l, r, series[m], 'min'))
            l = r - 1
            m = r
            r = r + 1

    return extremums


def get_extremums2d(image):

    extremums = []
    print(image.shape[1])
    for i in range(image.shape[1]):
        extremums.append(get_extremums(image[:, i]))

    return extremums

def get_extremums_with_filtration_2d(image):

    extremums = []

    for i in range(image.shape[0]):
        extremums.append(get_extremums(wavedec_filtration(image[i, :], [1, 1, 1, 0, 0, 0, 0, 0])))

    return extremums