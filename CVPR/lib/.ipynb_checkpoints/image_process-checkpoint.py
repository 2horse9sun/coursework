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
        return np.exp(-(x**2)/(2*std*std))

def guassian_pdf_2d(x, y, std):
        return np.exp(-(x**2+y**2)/(2*std*std))

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


def fourier_transform(img):
    f = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(f)
    return dft_shift

def inverse_fourier_transform(img_dft):
    ishift = np.fft.ifftshift(img_dft)
    iimg = np.fft.ifft2(ishift)
    return np.uint8(np.abs(iimg))

def generate_rect_hpf(img_dft, m, n):
    height, width = img_dft.shape
    hpf = np.ones((height, width))
    ch, cw = int(height/2), int(width/2)
    hpf[max(0,ch-m):min(ch+m,height-1), max(0,cw-n):min(cw+n,width-1)] = 0
    return hpf

def generate_circle_hpf(img_dft, r):
    height, width = img_dft.shape
    hpf = np.ones((height, width))
    ch, cw = int(height/2), int(width/2)
    for i in range(max(0,ch-r), min(ch+r,height-1)+1):
        for j in range(max(0,cw-r), min(cw+r,width-1)+1):
            if (i-ch)**2+(j-cw)**2 <= r**2:
                hpf[i][j] = 0
    return hpf

def generate_rect_lpf(img_dft, m, n):
    height, width = img_dft.shape
    lpf = np.zeros((height, width))
    ch, cw = int(height/2), int(width/2)
    lpf[max(0,ch-m):min(ch+m,height-1), max(0,cw-n):min(cw+n,width-1)] = 1
    return lpf

def generate_circle_lpf(img_dft, r):
    height, width = img_dft.shape
    lpf = np.zeros((height, width))
    ch, cw = int(height/2), int(width/2)
    for i in range(max(0,ch-r), min(ch+r,height-1)+1):
        for j in range(max(0,cw-r), min(cw+r,width-1)+1):
            if (i-ch)**2+(j-cw)**2 <= r**2:
                lpf[i][j] = 1
    return lpf

def subsample(img, factor_x, factor_y):
    height, width = img.shape
    new_height, new_width = height//factor_x, width//factor_y
    new_img = np.zeros((new_height, new_width))
    for i in range(0, new_height):
        for j in range(0, new_width):
            new_img[i][j] = img[min(int(i*factor_x), height-1)][min(int(j*factor_y), height-1)]
    return np.uint8(new_img)

def subsample_after_filtered(img, factor_x, factor_y):
    img_dft = fourier_transform(img)
    magnitude = np.abs(img_dft)
    phase = np.angle(img_dft)
    height = magnitude.shape[0]
    width = magnitude.shape[1]
    lpf = generate_rect_lpf(img_dft, height//2//factor_x, width//2//factor_y)
    magnitude = magnitude*lpf
    reconstructed_dft = magnitude*np.exp(np.complex(0,1)*phase)
    reconstructed_img = inverse_fourier_transform(reconstructed_dft)
    return subsample(reconstructed_img, factor_x, factor_y)