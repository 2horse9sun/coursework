import cv2
import matplotlib.pyplot as plt
import numpy as np

# show img read by cv2 using plt
def show_img(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
# normalize pixel values to 0-255
def normalize(img):
    normalized = np.zeros((img.shape[0],img.shape[1]))
    normalized = cv2.normalize(img,normalized,0,255,cv2.NORM_MINMAX)
    normalized = np.uint8(normalized)
    return normalized
