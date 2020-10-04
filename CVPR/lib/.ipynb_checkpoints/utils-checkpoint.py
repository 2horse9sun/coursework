import cv2
import matplotlib.pyplot as plt
import numpy as np

# show img read by cv2 using plt
def show_img(arr):
    plt.imshow(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))