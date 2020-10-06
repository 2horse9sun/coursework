# 1. Filter

Definition: Form a new image whose pixels are a combination of the original pixels.

–	To get useful information from images (extract edges or contours (to understand shape))

–	To enhance the image (to blur to remove noise, to sharpen to “enhance image”)

# 2. Mean Filter

Definition: Replace pixel by mean of neighborhood.

General expression:
$$
f(m,n)=\sum_{i=-k}^{k}\sum_{j=-k}^{k}w(i,j)f(m+i,n+j)
$$

$$
w=\frac{1}{(2k+1)^2}
\begin{bmatrix}
1&1&\cdots&1\\
1&1&\cdots&1\\
\vdots&\vdots&\ddots&1\\
1&1&\cdots&1\\
\end{bmatrix},aka.w(i,j)=\frac{1}{(2k+1)^2}
$$

## 2.1 Example 01

