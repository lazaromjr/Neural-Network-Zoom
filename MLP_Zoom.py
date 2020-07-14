from PIL import Image
from zoom import *
from psnr import psnr
import time

print("MLP Zoom")
print("--------")

path = 'teste1'

img = Image.open(path + '/original100p.png')

img2 = annZoom(img, 2, 5)
img2.save(path + '/NeuralNetwork2x.png')

img3 = annZoom(img, 4, 5)
img3.save(path + '/NeuralNetwork4x.png')
img3.save('test/NeuralNetwork4x.png')

img4 = nearestZoom(img, 2)
img4.save(path + '/nearest2x.png')

img5 = nearestZoom(img, 4)
img5.save(path + '/nearest4x.png')

img6 = bilinearZoom(img, 2)
img6.save(path + '/bilinear2x.png')

img7 = bilinearZoom(img, 4)
img7.save(path + '/bilinear4x.png')



#psnrVar = psnr(imgFull, img2)
#print(psnrVar)