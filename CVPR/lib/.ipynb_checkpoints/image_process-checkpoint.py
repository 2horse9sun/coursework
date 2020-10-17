##################################################################
# Naive implementation of some techniques in image processing
# @Auther 2horse9sun
##################################################################
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

##################################################################
# Filtering
##################################################################
def mean_filter(img, k):
    # mean filter kernel
    w = np.ones((2*k+1, 2*k+1))
    w = w / w.sum()
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

def median_filter(img, k):
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
        reflect_img[0*height:(0+1)*height, 0*width:(0+1)*width] = np.fliplr(np.flipud(img))
        reflect_img[0*height:(0+1)*height, 1*width:(1+1)*width] = np.flipud(img)
        reflect_img[0*height:(0+1)*height, 2*width:(2+1)*width] = np.fliplr(np.flipud(img))
        reflect_img[1*height:(1+1)*height, 0*width:(0+1)*width] = np.fliplr(img)
        reflect_img[1*height:(1+1)*height, 1*width:(1+1)*width] = img
        reflect_img[1*height:(1+1)*height, 2*width:(2+1)*width] = np.fliplr(img)
        reflect_img[2*height:(2+1)*height, 0*width:(0+1)*width] = np.fliplr(np.flipud(img))
        reflect_img[2*height:(2+1)*height, 1*width:(1+1)*width] = np.flipud(img)
        reflect_img[2*height:(2+1)*height, 2*width:(2+1)*width] = np.fliplr(np.flipud(img))
        new_img = reflect_img[height-dh:2*height+dh, width-dw:2*width+dw]
        return np.uint8(new_img)
    
            


def gaussian_pdf_1d(x, sigma):
    if sigma == 0:
        return 1 if x==0 else 0
    return np.exp(-(x**2)/(2*sigma*sigma))

def gaussian_pdf_2d(x, y, sigma):
    if sigma == 0:
        return 1 if x==0 and y==0 else 0
    return np.exp(-(x**2+y**2)/(2*sigma*sigma))
    
def generate_gaussian_kernel_1d(k, sigma):
    kernel = np.zeros((2*k+1, 1))
    for i in range(-k, k+1):
        kernel[i+k] = gaussian_pdf_1d(i, sigma)
    return kernel/kernel.sum()
    
def generate_gaussian_kernel_2d(k, sigma):
    return generate_gaussian_kernel_1d(k, sigma).dot(generate_gaussian_kernel_1d(k, sigma).T)

def convolve_2d(w, f, m, n):
    res = 0
    k = w.shape[0]//2
    l = w.shape[1]//2
    for i in range(-k, k+1):
        for j in range(-l, l+1):
            res += w[i+k][j+l]*f[m-i][n-j]
    return res


# default boundary condition: valid convolution
def gaussian_filter(img, k, sigma):
    w = generate_gaussian_kernel_2d(k, sigma)    
    height = img.shape[0]
    width = img.shape[1]
    img_gaussian_filtered = np.zeros((height, width))

    for m in range(0, height):
        for n in range(0, width):
            if m < k or n < k or m >= height-k or n >= width-k:
                img_gaussian_filtered[m][n] = img[m][n]
            else:
                img_gaussian_filtered[m][n] = convolve_2d(w, img, m, n)
    img_gaussian_filtered = np.uint8(img_gaussian_filtered)
    return img_gaussian_filtered[k:height-k, k:width-k]

def separable_gaussian_filter(img, k, sigma):
    u = generate_gaussian_kernel_1d(k, sigma).T    
    v = generate_gaussian_kernel_1d(k, sigma)   
    height = img.shape[0]
    width = img.shape[1]
    img_gaussian_filtered = np.zeros((height, width))
    img_gaussian_row_filtered = np.zeros((height, width))

    for m in range(0, height):
        for n in range(0, width):
            if m < k or n < k or m >= height-k or n >= width-k:
                img_gaussian_row_filtered[m][n] = img[m][n]
            else:
                img_gaussian_row_filtered[m][n] = convolve_2d(v, img, m, n)
    for m in range(0, height):
        for n in range(0, width):
            if m < k or n < k or m >= height-k or n >= width-k:
                img_gaussian_filtered[m][n] = img_gaussian_row_filtered[m][n]
            else:
                img_gaussian_filtered[m][n] = convolve_2d(u, img_gaussian_row_filtered, m, n)
    img_gaussian_filtered = np.uint8(img_gaussian_filtered)
    return img_gaussian_filtered[k:height-k, k:width-k]

def sharpening_filter(img, k, sigma, alpha):
    # f = f + a(f - f_blur)
    return np.uint8((1+alpha)*separable_gaussian_filter(img,k,0) - alpha*separable_gaussian_filter(img, k, sigma))

def bilateral_filter(img, k, sigma_s, sigma_r):
    height = img.shape[0]
    width = img.shape[1]
    img_bilateral_filtered = np.zeros((height, width))

    for m in range(0, height):
        for n in range(0, width):
            if m < k or n < k or m >= height-k or n >= width-k:
                img_bilateral_filtered[m][n] = img[m][n]
            else:
                w = np.zeros((2*k+1, 2*k+1))
                for i in range(-k, k+1):
                    for j in range(-k, k+1):
                        w[i+k][j+k] = gaussian_pdf_2d(i, j, sigma_s) * gaussian_pdf_1d(img[m][n]-img[m+i][n+j], sigma_r)
                w = w / w.sum()
                img_bilateral_filtered[m][n] = convolve_2d(w, img, m, n)
    img_bilateral_filtered = np.uint8(img_bilateral_filtered)
    return img_bilateral_filtered[k:height-k, k:width-k]


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
    return 1 - generate_rect_hpf(img_dft, m, n)

def generate_circle_lpf(img_dft, r):
    return 1 - generate_circle_hpf(img_dft, r)

# H(u, v) = exp{-D(u,v)^2/2*D0^2}
def generate_gaussian_lpf(img_dft, D0):
    height, width = img_dft.shape
    lpf = np.zeros((height, width))
    ch, cw = int(height/2), int(width/2)
    for i in range(0, height):
        for j in range(0, width):
            lpf[i][j] = gaussian_pdf_2d(i-ch, j-cw, D0)
    return lpf

def generate_gaussian_hpf(img_dft, D0):
    return 1 - generate_gaussian_lpf(img_dft, D0)

##################################################################
# Transform
##################################################################
# subsample the image directly
def subsample(img):
    height, width = img.shape
    new_height, new_width = height//2, width//2
    new_img = np.zeros((new_height, new_width))
    for i in range(0, new_height):
        for j in range(0, new_width):
            new_img[i][j] = img[min(int(i*2), height)][min(int(j*2), height)]
    return np.uint8(new_img)

# 1. blur the image by lpf
# 2. subsample the filtered image
def subsample_after_filtered(img):
    img_filtered = cv2.GaussianBlur(img,(5,5),1.0825)
    return subsample(img_filtered)

# upsample the image directly, fill with 0s
def upsample(img):
    height, width = img.shape
    new_height, new_width = height*2, width*2
    new_img = np.zeros((new_height, new_width))
    for i in range(0, height):
        for j in range(0, width):
            new_img[min(i*2, new_height)][min(j*2,new_width)] = img[i][j]
    return np.uint8(new_img)

# 1. upsample the image directly, fill with 0s
# 2. blur the image (one technique of interpolation)
# 3. scale up the image
def upsample_then_filtered(img):
    img_upsampled = upsample(img)
    img_filtered = cv2.GaussianBlur(img_upsampled,(5,5),1.0825)
    img_filtered = img_filtered*2*2
    return img_filtered

def get_interpolation_value(img, x, y, type):
    if int(x)==x and int(y)==y:
        return img[int(x)][int(y)]
    if type == 'nearest':
        return img[min(math.floor(x+0.5), img.shape[0]-1)][min(math.floor(y+0.5), img.shape[1]-1)]
    elif type == 'bilinear':
        x0 = min(math.floor(x), img.shape[0]-1)
        y0 = min(math.floor(y), img.shape[1]-1)
        x1 = min(math.floor(x+1), img.shape[0]-1)
        y1 = min(math.floor(y+1), img.shape[1]-1)
        S11 = (x1-x)*(y1-y)
        S10 = (x-x0)*(y1-y)
        S01 = (x1-x)*(y-y0)
        S00 = (x-x0)*(y-y0)
        S = S11 + S10 + S01 + S00
        if S==0:
            return img[int(x)][int(y)]
        return (S11*img[x0][y0]+S10*img[x0][y1]+S01*img[x1][y0]+S00*img[x1][y1])/S
    else:
        return 0
    
def upsample_with_interpolation(img, factor, type):
    height, width = img.shape
    new_height, new_width = int(height*factor), int(width*factor)
    new_img = np.zeros((new_height, new_width))
    for i in range(0, new_height):
        for j in range(0, new_width):
            new_img[i][j] = get_interpolation_value(img, i/factor, j/factor, type)
    return np.uint8(new_img)


def get_translation_kernel(x, y):
    return np.array([[1,0,x],
                    [0,1,y],
                    [0,0,1]])
def get_rotation_kernel(theta):
    theta = theta*np.pi/180
    return np.array([[np.cos(theta),-np.sin(theta),0],
                    [np.sin(theta),np.cos(theta),0],
                    [0,0,1]])
def get_euclidean_kernel(x, y, theta):
    theta = theta*np.pi/180
    return np.array([[np.cos(theta),-np.sin(theta),x],
                    [np.sin(theta),np.cos(theta),y],
                    [0,0,1]])
def get_similarity_kernel(x, y, theta, s):
    theta = theta*np.pi/180
    return np.array([[s*np.cos(theta),-s*np.sin(theta),x],
                    [s*np.sin(theta),s*np.cos(theta),y],
                    [0,0,1]])

def forward_warping(img, kernel):
    height = img.shape[0]
    width = img.shape[1]
    img_transformed = np.zeros((height, width))
    for m in range(0, height):
        for n in range(0, width):
            pos = kernel.dot(np.array([[m],
                                      [n],
                                      [1]]))
            mm = int(pos[0][0])
            nn = int(pos[1][0])
            if mm>=0 and mm<height and nn>=0 and nn<width:
                img_transformed[mm][nn] = img[m][n]
    return np.uint8(img_transformed)
    
    
def inverse_warping(img, kernel):
    height = img.shape[0]
    width = img.shape[1]
    img_transformed = np.zeros((height, width))
    for m in range(0, height):
        for n in range(0, width):
            pos = np.linalg.inv(kernel).dot(np.array([[m],
                                      [n],
                                      [1]]))
            mm = pos[0][0]
            nn = pos[1][0]
            if mm>=0 and mm<height and nn>=0 and nn<width:
                img_transformed[m][n] = get_interpolation_value(img, mm, nn, 'bilinear')
    return np.uint8(img_transformed)
    


##################################################################
# Feature Detection
##################################################################
def get_sobel_x():
    return np.array([[-1, 0, 1]]).T, np.array([[1, 2, 1]])
def get_sobel_y():
    return np.array([[1, 2, 1]]).T, np.array([[-1, 0, 1]])

# sobel-based
def gradient(img):
    k = 1
    u_x, v_x = get_sobel_x()   
    u_y, v_y = get_sobel_y()  
    height = img.shape[0]
    width = img.shape[1]
    gradient_x = np.zeros((height, width))
    gradient_x_row = np.zeros((height, width))
    gradient_y = np.zeros((height, width))
    gradient_y_row = np.zeros((height, width))

    for m in range(0, height):
        for n in range(0, width):
            if m < k or n < k or m >= height-k or n >= width-k:
                gradient_x_row[m][n] = 0
            else:
                gradient_x_row[m][n] = convolve_2d(v_x, img, m, n)
                
    for m in range(0, height):
        for n in range(0, width):
            if m < k or n < k or m >= height-k or n >= width-k:
                gradient_x[m][n] = gradient_x_row[m][n]
            else:
                gradient_x[m][n] = convolve_2d(u_x, gradient_x_row, m, n)
    
    for m in range(0, height):
        for n in range(0, width):
            if m < k or n < k or m >= height-k or n >= width-k:
                gradient_y_row[m][n] = 0
            else:
                gradient_y_row[m][n] = convolve_2d(v_y, img, m, n)
    for m in range(0, height):
        for n in range(0, width):
            if m < k or n < k or m >= height-k or n >= width-k:
                gradient_y[m][n] = gradient_y_row[m][n]
            else:
                gradient_y[m][n] = convolve_2d(u_y, gradient_y_row, m, n)
    
    return gradient_x[k:height-k, k:width-k], gradient_y[k:height-k, k:width-k]

# non-maximum supression, gradient
def nms(gradX, gradY):
    magnitude = np.sqrt(gradX[:, :]**2 + gradY[:, :]**2)
    height = gradX.shape[0]
    width = gradX.shape[1]
    nms = np.zeros((height, width))
    
    for m in range(1, height-1):
        for n in range(1, width-1):
            q = [m, n]
            q = np.array(q)
            q_gradient = [gradX[m][n], gradY[m][n]]
            q_magnitude = magnitude[m][n]
            if q_magnitude==0:
                continue
            r = q + q_gradient/q_magnitude
            p = q - q_gradient/q_magnitude
            r_magnitude = get_interpolation_value(magnitude, r[0], r[1], 'bilinear')
            p_magnitude = get_interpolation_value(magnitude, p[0], p[1], 'bilinear')
            # find local maximum
            if q_magnitude>r_magnitude and q_magnitude>p_magnitude:
                nms[m][n] = magnitude[m][n]
            else:
                nms[m][n] = 0
    return nms

def connect_edges(low, high, visited, m, n):
    if high[m][n] == 0 or low[m][n] == 0:
        return
    dx = [-1, 0, 1]
    dy = [-1, 0, 1]
    queue = []
    queue.append([m, n])
    while len(queue) != 0:
        curr = queue.pop()
        x = curr[0]
        y = curr[1]
        high[x][y] = 255
        visited[x][y] = 1
        for i in range(0, 3):
            for j in range(0, 3):
                # weak edges
                if x+dx[i]>=0 and x+dx[i]<low.shape[0] and y+dy[j]>=0 and y+dy[j]<low.shape[1]:
                    if visited[x+dx[i]][y+dy[j]]!=1 and low[x+dx[i]][y+dy[j]]!=0:
                        queue.append([x+dx[i], y+dy[j]])
                        
def hysteresis_thresholding(img, tl, th):
    _, low = cv2.threshold(img, tl, 255, cv2.THRESH_BINARY)
    _, high = cv2.threshold(img, th, 255, cv2.THRESH_BINARY)
    height = high.shape[0]
    width = high.shape[1]
    visited = np.zeros((height, width))
    for m in range(0, height):
        for n in range(0, width):
            connect_edges(low, high, visited, m, n)
    return high


def compute_second_moment_matrix(img, k, sigma):
    # compute gradients Ix, Iy
    Ix = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=1, dy=0)
    Iy = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=0, dy=1)
    # compute Ix2, Iy2, Ixy
    Ix2 = Ix*Ix
    Iy2 = Iy*Iy
    Ixy = Ix*Iy
    # convovle with window function: Gaussian
    g_Ix2 = cv2.GaussianBlur(Ix2,(2*k+1, 2*k+1),sigma)
    g_Iy2 = cv2.GaussianBlur(Iy2,(2*k+1, 2*k+1),sigma)
    g_Ixy = cv2.GaussianBlur(Ixy,(2*k+1, 2*k+1),sigma)
    # assemble matrix at each pixel
    height = img.shape[0]
    width = img.shape[1]
    smm_map = np.zeros((height, width, 2, 2))
    for m in range(0, height):
        for n in range(0, width):
            smm_map[m][n] = np.array([[g_Ix2[m][n], g_Ixy[m][n]],
                                    [g_Ixy[m][n], g_Iy2[m][n]]])
    return smm_map

# non-maximum supression, 2d
def nms_2d(img, threshold):
    dx = [-1, 0, 1]
    dy = [-1, 0, 1]
    height = img.shape[0]
    width = img.shape[1]
    nms = np.zeros((height, width))
    
    for m in range(1, height-1):
        for n in range(1, width-1):
            center = img[m][n]
            neighbours = []
            for i in range(0, 3):
                for j in range(0, 3):
                    if m+dx[i]>=0 and m+dx[i]<height and n+dy[j]>=0 and n+dy[j]<width:
                        neighbours.append(img[m+dx[i]][n+dy[j]])
            if center == max(neighbours) and center > threshold:
                nms[m][n] = 255
    return np.uint8(nms)