import numpy as np
import cv2 as cv
import pywt


def apply_dwt(image, wname='db6'):
    A, (B, C, D) = pywt.dwt2(image, wname)
    na, ma = A.shape
    nb, mb = B.shape
    nc, mc = C.shape
    nd, md = D.shape
    new_image = np.zeros((na + nc, ma + mb))
    new_image[:na, :ma] = A
    new_image[:nb, ma:] = B
    new_image[na:, :mc] = C
    new_image[na:, ma:] = D
    return new_image, A, B, C, D


def apply_dwt_n(image, wname='db6'):
    new_image, A, B, C, D = apply_dwt(image,wname)
    A, AA, BB, CC, DD = apply_dwt(A, wname)
    new_image[:A.shape[0], :A.shape[1]] = A
    return new_image, A, B, C, D


cap = cv.VideoCapture(0)

while True:
    image, ret = cap.read()

    image = cv.cvtColor(ret, cv.COLOR_BGR2GRAY)

    image = image / 256

    wname = 'db1'

    cv.imshow('video', image)
    new_image, A, B, C, D = apply_dwt(image, wname)
    cv.imshow('all', new_image)

    # B.fill(0)
    # C.fill(0)
    # D.fill(0)

    cv.imshow('rec image', pywt.idwt2((A, (B, C, D)), wname))

    if cv.waitKey(1) == 113:
        break

cv.destroyAllWindows()