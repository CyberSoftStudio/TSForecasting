import numpy as np
import matplotlib.pyplot as plt
import pywt

cat = plt.imread('castle.png')
image = 0.3 * cat[:,:,0] + 0.3 * cat[:,:,1] + 0.3 * cat[:,:,2]

n, m = image.shape
plt.figure()
plt.imshow(image,cmap='gray')

wname = 'db16'

A, (B, C, D) = pywt.dwt2(image, wname)
plt.figure()
plt.subplot(221)
plt.imshow(A, cmap='gray')
plt.subplot(222)
plt.imshow(B, cmap='gray')
plt.subplot(223)
plt.imshow(C, cmap='gray')
plt.subplot(224)
plt.imshow(D, cmap='gray')
#
#
# A.fill(0)
B.fill(0)
C.fill(0)
D.fill(0)

rec_image = pywt.idwt2((A, (B, C, D)), wname)

mse = 1/(n * m) * np.sum((image - rec_image) ** 2)
psnr = 10 * np.log10(1/mse)

print(psnr)

plt.figure()

plt.imshow(rec_image, cmap='gray')
plt.show()