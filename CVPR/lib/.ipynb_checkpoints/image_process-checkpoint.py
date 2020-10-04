import cv2
import matplotlib.pyplot as plt
import numpy as np

def mean_filter(img, size):
    # mean filter kernel
    w = np.ones((size, size))
    w = w / w.sum()
    k = size // 2
    height = img.shape[0]
    width = img.shape[1]
    img_mean_filtered = np.zeros((height, width))

    # mean filter
    for m in range(0, height):
        for n in range(0, width):
            if m < k or n < k or m >= height-k or n >= width-k:
                img_mean_filtered[m][n] = img[m][n]
            else:
                for i in range(-k, k+1):
                    for j in range(-k, k+1):
                        img_mean_filtered[m][n] += w[i+k][j+k]*img[m+i][n+j]
    img_mean_filtered = np.uint8(img_mean_filtered)
    return img_mean_filtered


def sharpening_filter(img, size, alpha):
    return np.uint8(img + alpha*(img-mean_filter(img, size)))

def thresholding_filter(img, A):
    height = img.shape[0]
    width = img.shape[1]
    img_mean_filtered = np.zeros((height, width))

    for m in range(0, height):
        for n in range(0, width):
            if img[m][n] > A:
                img_mean_filtered[m][n] = 255
            else:
                img_mean_filtered[m][n] = 0
            
    img_mean_filtered = np.uint8(img_mean_filtered)
    return img_mean_filtered

def median_filter(img, size):
    k = size // 2
    height = img.shape[0]
    width = img.shape[1]
    img_median_filtered = np.zeros((height, width))

    for m in range(0, height):
        for n in range(0, width):
            if m < k or n < k or m >= height-k or n >= width-k:
                img_median_filtered[m][n] = img[m][n]
            else:
                img_median_filtered[m][n] = np.median(img[m-k:m+k,n-k:n+k])
    img_median_filtered = np.uint8(img_median_filtered)
    return img_median_filtered


def guassian_pdf_1d(x, std):
        return np.exp(-(x^2)/(2*std*std))

def guassian_pdf_2d(x, y, std):
        return np.exp(-(x^2+y^2)/(2*std*std))

def guassian_filter(img, size, std):
    k = size // 2
    
    w = np.zeros((size, size))
    for i in range(-k, k+1):
        for j in range(-k, k+1):
            w[i+k][j+k] = guassian_pdf_2d(i, j, std)
    w = w / w.sum()
    
    height = img.shape[0]
    width = img.shape[1]
    img_guassian_filtered = np.zeros((height, width))

    for m in range(0, height):
        for n in range(0, width):
            if m < k or n < k or m >= height-k or n >= width-k:
                img_guassian_filtered[m][n] = img[m][n]
            else:
                for i in range(-k, k+1):
                    for j in range(-k, k+1):
                        img_guassian_filtered[m][n] += w[i+k][j+k]*img[m+i][n+j]
    img_guassian_filtered = np.uint8(img_guassian_filtered)
    return img_guassian_filtered

def bilateral_filter(img, size, std_s, std_r):
    k = size // 2
    
    height = img.shape[0]
    width = img.shape[1]
    img_bilateral_filtered = np.zeros((height, width))

    for m in range(0, height):
        for n in range(0, width):
            if m < k or n < k or m >= height-k or n >= width-k:
                img_bilateral_filtered[m][n] = img[m][n]
            else:
                w = np.zeros((size, size))
                for i in range(-k, k+1):
                    for j in range(-k, k+1):
                        w[i+k][j+k] = guassian_pdf_2d(i, j, std_s) * guassian_pdf_1d(img[m][n]-img[m+i][n+j], std_r)
                w = w / w.sum()
                
                for i in range(-k, k+1):
                    for j in range(-k, k+1):
                        img_bilateral_filtered[m][n] += w[i+k][j+k]*img[m+i][n+j]
    img_bilateral_filtered = np.uint8(img_bilateral_filtered)
    return img_bilateral_filtered