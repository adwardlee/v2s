import cv2
import numpy as np
from PIL import Image

height = 256
width = 340
a = np.zeros((height,width,3), np.uint8)
a[:,:,0] = 104
a[:,:,1] = 117
a[:,:,2] = 123
cv2.imwrite('mean.png',a)
