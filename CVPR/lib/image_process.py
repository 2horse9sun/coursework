##################################################################
# Naive implementation of some techniques in image processing
# @Auther 2horse9sun
##################################################################
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
    # f = f + a(f - f_blur)
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
                img_median_filtered[m][n] = np.median(img[m-k:m+k+1,n-k:n+k+1])
    img_median_filtered = np.uint8(img_median_filtered)
    return img_median_filtered


# valid boundary conditions: full, same, valid
# valid padding methods: clip, wrap, copy, reflect
# return the processed image
def boundary_process(img, k, cond, padding):
    height = img.shape[0]
    width = img.shape[1]
    # boundary conditions
    if cond == 'full':
        new_img = np.zeros((height+4*k, width+4*k))
        dh, dw = 2*k, 2*k
    elif cond == 'same':
        new_img = np.zeros((height+2*k, width+2*k))
        dh, dw = k, k
    elif cond == 'valid':
        return img
    else:
        return img
    
    new_height = new_img.shape[0]
    new_width = new_img.shape[1]
    
    # padding methods
    if padding == 'clip':
        for i in range(0, new_height):
            for j in range(0, new_width):
                if i>=dh and i<=new_height-1-dh and j>=dw and j<=new_width-1-dw:
                    new_img[i][j] = img[i-dh][j-dw]
                else:
                    new_img[i][j] = 0
        return np.uint8(new_img)
    if padding == 'wrap':
        wrap_img = np.zeros((3*height, 3*width))
        for i in range(0, 3):
            for j in range(0, 3):
                wrap_img[i*height:(i+1)*height, j*width:(j+1)*width] = img
        new_img = wrap_img[height-dh:2*height+dh, width-dw:2*width+dw]
        return np.uint8(new_img)
    if padding == 'copy':
        for i in range(0, new_height):
            for j in range(0, new_width):
                if i>=dh and i<=new_height-1-dh and j>=dw and j<=new_width-1-dw:
                    new_img[i][j] = img[i-dh][j-dw]
                elif i<dh or i>new_height-1-dh:
                    row = dh if i<dh else new_height-1-dh
                    if j < dw:
                        col = dw
                    elif j > new_width-1-dw:
                        col = new_width-1-dw
                    else:
                        col = j
                    new_img[i][j] = img[row-dh][col-dw]
                else:
                    col = dw if i<dh else new_width-1-dw
                    if j < dw:
                        col = dw
                    elif j > new_width-1-dw:
                        col = new_width-1-dw
                    new_img[i][j] = img[i-dh][col-dw]
        return np.uint8(new_img)
    if padding == 'reflect':
        reflect_img = np.zeros((3*height, 3*width))
        reflect_img[0*height:(0+1)*height, 0*width:(0+1)*width] = img.T
        reflect_img[0*height:(0+1)*height, 1*width:(1+1)*width] = np.flipud(img)
        reflect_img[0*height:(0+1)*height, 2*width:(2+1)*width] = img.T
        reflect_img[1*height:(1+1)*height, 0*width:(0+1)*width] = np.fliplr(img)
        reflect_img[1*height:(1+1)*height, 1*width:(1+1)*width] = img
        reflect_img[1*height:(1+1)*height, 2*width:(2+1)*width] = np.fliplr(img)
        reflect_img[2*height:(2+1)*height, 0*width:(0+1)*width] = img.T
        reflect_img[2*height:(2+1)*height, 1*width:(1+1)*width] = np.flipud(img)
        reflect_img[2*height:(2+1)*height, 2*width:(2+1)*width] = img.T
        new_img = reflect_img[height-dh:2*height+dh, width-dw:2*width+dw]
        return np.uint8(new_img)
    
            


def gaussian_pdf_1d(x, std):
        return np.exp(-(x**2)/(2*std*std))

def gaussian_pdf_2d(x, y, std):
        return np.exp(-(x**2+y**2)/(2*std*std))
    
def generate_gaussian_kernel_1d(k, std):
    kernel = np.zeros((k, 1))
    for i in range(-k, k+1):
        kernel[i+k] = image_process.gaussian_pdf_2d(i, std)
    return kernel/kernel.sum()
    
def generate_gaussian_kernel_2d(k, std):
    return generate_gaussian_kernel_1d(2*k+1, std).dot(generate_gaussian_kernel_1d(2*k+1, std).T)


# default boundary condition: valid convolution
def gaussian_filter(img, size, std):
    k = size // 2
    w = np.zeros((size, size))
    for i in range(-k, k+1):
        for j in range(-k, k+1):
            w[i+k][j+k] = gaussian_pdf_2d(i, j, std)
    w = w / w.sum()
    
    height = img.shape[0]
    width = img.shape[1]
    img_gaussian_filtered = np.zeros((height, width))

    for m in range(0, height):
        for n in range(0, width):
            if m < k or n < k or m >= height-k or n >= width-k:
                img_gaussian_filtered[m][n] = img[m][n]
            else:
                for i in range(-k, k+1):
                    for j in range(-k, k+1):
                        img_gaussian_filtered[m][n] += w[i+k][j+k]*img[m+i][n+j]
    img_gaussian_filtered = np.uint8(img_gaussian_filtered)
    return img_gaussian_filtered[k:height-k, k:width-k]

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
                        w[i+k][j+k] = gaussian_pdf_2d(i, j, std_s) * gaussian_pdf_1d(img[m][n]-img[m+i][n+j], std_r)
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

# high-pass filter
# Example
# 1 1 1 1 1 1 1
# 1 1 1 1 1 1 1
# 1 1 0 0 0 1 1
# 1 1 0 0 0 1 1
# 1 1 0 0 0 1 1
# 1 1 1 1 1 1 1
# 1 1 1 1 1 1 1
def generate_rect_hpf(img_dft, m, n):
    height, width = img_dft.shape
    hpf = np.ones((height, width))
    ch, cw = int(height/2), int(width/2)
    hpf[max(0,ch-m):min(ch+m,height), max(0,cw-n):min(cw+n,width)] = 0
    return hpf

def generate_circle_hpf(img_dft, r):
    height, width = img_dft.shape
    hpf = np.ones((height, width))
    ch, cw = int(height/2), int(width/2)
    for i in range(max(0,ch-r), min(ch+r,height)):
        for j in range(max(0,cw-r), min(cw+r,width)):
            if (i-ch)**2+(j-cw)**2 <= r**2:
                hpf[i][j] = 0
    return hpf

# low-pass filter
# Example
# 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0
# 0 0 1 1 1 0 0
# 0 0 1 1 1 0 0
# 0 0 1 1 1 0 0
# 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0
def generate_rect_lpf(img_dft, m, n):
    height, width = img_dft.shape
    lpf = np.zeros((height, width))
    ch, cw = int(height/2), int(width/2)
    lpf[max(0,ch-m):min(ch+m,height), max(0,cw-n):min(cw+n,width)] = 1
    return lpf

def generate_circle_lpf(img_dft, r):
    height, width = img_dft.shape
    lpf = np.zeros((height, width))
    ch, cw = int(height/2), int(width/2)
    for i in range(max(0,ch-r), min(ch+r,height)):
        for j in range(max(0,cw-r), min(cw+r,width)):
            if (i-ch)**2+(j-cw)**2 <= r**2:
                lpf[i][j] = 1
    return lpf

# subsample the image directly
def subsample(img, factor_x, factor_y):
    height, width = img.shape
    new_height, new_width = height//factor_x, width//factor_y
    new_img = np.zeros((new_height, new_width))
    for i in range(0, new_height):
        for j in range(0, new_width):
            new_img[i][j] = img[min(int(i*factor_x), height)][min(int(j*factor_y), height)]
    return np.uint8(new_img)

# 1. blur the image by lpf
# 2. subsample the filtered image
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

# upsample the image directly, fill with 0s
def upsample(img, factor_x, factor_y):
    height, width = img.shape
    new_height, new_width = height*factor_x, width*factor_y
    new_img = np.zeros((new_height, new_width))
    for i in range(0, height):
        for j in range(0, width):
            new_img[min(i*factor_x, new_height)][min(j*factor_y,new_width)] = img[i][j]
    return np.uint8(new_img)

# 1. upsample the image directly, fill with 0s
# 2. blur the image (one technique of interpolation)
# 3. scale up the image
def upsample_then_filtered(img, factor_x, factor_y):
    img_upsampled = upsample(img, factor_x, factor_y)
    img_dft = fourier_transform(img_upsampled)
    magnitude = np.abs(img_dft)
    phase = np.angle(img_dft)
    height = magnitude.shape[0]
    width = magnitude.shape[1]
    lpf = generate_rect_lpf(img_dft, height//2//factor_x, width//2//factor_y)
    magnitude = magnitude*lpf
    reconstructed_dft = magnitude*np.exp(np.complex(0,1)*phase)
    reconstructed_img = inverse_fourier_transform(reconstructed_dft)
    reconstructed_img = reconstructed_img*factor_x*factor_y
    return reconstructed_img
