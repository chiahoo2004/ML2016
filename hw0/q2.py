from scipy import ndimage
from scipy import misc

lena = misc.imread('Lena.png')
rotate_lena = ndimage.rotate(lena, 180)
misc.imsave("ans2.png", rotate_lena);
