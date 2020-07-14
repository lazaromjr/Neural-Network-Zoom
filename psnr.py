from math import log10, sqrt
import numpy as np 

def psnr(originalImg, compressedImg):

    #escala de cinza
    originalImg = originalImg.convert('L')
    compressedImg = compressedImg.convert('L')

    #converte imagens em array
    original = np.asarray(originalImg, dtype=np.int)
    compressed = np.asarray(compressedImg, dtype=np.int)

    mse = np.mean((original - compressed) ** 2)

    #Se não há ruído
    if (mse == 0):
        return 100

    max_pixel = 255.0
    psnr = 10 * log10(max_pixel ** 2 / mse)
    #psnr = 20 * log10(max_pixel / sqrt(mse))
    

    return psnr