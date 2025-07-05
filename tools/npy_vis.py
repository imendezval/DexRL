import numpy as np

from matplotlib import pyplot as plt
import matplotlib.image
from PIL import Image
import cv2


img_array = np.load('depth_img.npy')

mask = np.zeros((386, 516), dtype=float)
mask[110:275, 130:380] = 1

mask_uint8 = (mask * 255).astype(np.uint8)
Image.fromarray(mask_uint8).save('segmask.png')


# board_mod = np.where(mask > 0, img_array, 0)
# plt.imshow(board_mod, cmap='gray')
# plt.show()