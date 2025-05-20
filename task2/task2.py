import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 1. Prepare the image ------------------------------------------------------------------------------------------------------------------------------------------
image_pil = Image.open('/Users/cristiad/Documents/Master Degree/1st Semester/Image Analysis/iaor/task2/ampelmaennchen.png').convert("L") 
image = np.array(image_pil, dtype=np.float32) / 255.0

# Step a: Define Kernel Gaussian
def create_gaussian_kernel(size, sigma):
    center = size // 2
    kernel = np.zeros(size)
    for i in range(size):
        x = i - center
        kernel[i] = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(x**2) / (2 * sigma**2))
    return kernel

def create_gaussian_derivative_kernel(size, sigma):
    center = size // 2
    kernel = np.zeros(size)
    for i in range(size):
        x = i - center
        kernel[i] = -x * (1.0 / (np.sqrt(2 * np.pi) * sigma**3)) * np.exp(-(x**2) / (2 * sigma**2))
    return kernel

# Kernel parameter
size = 11
sigma = 1.5

# Make 2D Kernel fro Outer Products
G = create_gaussian_kernel(size, sigma)
G_deriv = create_gaussian_derivative_kernel(size, sigma)
GoGx = np.outer(G_deriv, G)
GoGy = np.outer(G, G_deriv)

# Step b: Manual Convolution
def manual_convolve2d(image, kernel):
    i_h, i_w = image.shape
    k_h, k_w = kernel.shape
    pad_h = k_h // 2
    pad_w = k_w // 2
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output = np.zeros_like(image)
    for i in range(i_h):
        for j in range(i_w):
            region = padded_image[i:i + k_h, j:j + k_w]
            output[i, j] = np.sum(region * kernel)
    return output

# GoG Filter application
Ix = manual_convolve2d(image, GoGx)
Iy = manual_convolve2d(image, GoGy)

# Step c: Calculate Gradien Magnitude
grad_magnitude = np.sqrt(Ix**2 + Iy**2)

# Show results
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(Ix, cmap='gray')
plt.title('Gradient Ix (GoGx)')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(Iy, cmap='gray')
plt.title('Gradient Iy (GoGy)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(grad_magnitude, cmap='gray')
plt.title('Gradient Magnitude')
plt.axis('off')

plt.tight_layout()
plt.show()