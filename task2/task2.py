import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 1. Prepare the image --------------------------------------------
image_pil = Image.open('ampelmaennchen.png').convert("L")
image_array = np.array(image_pil, dtype=np.float32) / 255.0

# -------------------- GRAYSCALE & ENHANCEMENT --------------------
# Convert to Grayscale manually
grayscale_array = (image_array * 255).astype(np.uint8)


# Contrast stretching (enhancement)
min_intensity = grayscale_array.min()
max_intensity = grayscale_array.max()
enhanced_grayscale_array = ((grayscale_array - min_intensity) / (max_intensity - min_intensity)) * 255
enhanced_grayscale_array_rounded = np.floor(enhanced_grayscale_array).astype(np.uint8)

image = enhanced_grayscale_array_rounded.astype(np.float32) / 255.0

def create_gaussian_derivative_kernel(xSize, ySize, sigma):
    xCenter = xSize // 2
    yCenter = ySize // 2
    kernel = np.zeros((xSize, ySize))
    for i in range(xSize):
        x = i - xCenter
        for j in range(ySize):
            y = j - yCenter
            kernel[i,j] = (-x/ ((2 * np.pi) * sigma**4)) * np.exp(-(((x**2) + (y**2)) / (2 * sigma**2)))
    return kernel

# Kernel Parameters
SizeX = 5
SizeY = 5
Sigma = 0.5

Gx_deriv = create_gaussian_derivative_kernel(SizeX, SizeY, Sigma)
print(Gx_deriv)
Gy_deriv = Gx_deriv.T
print(Gy_deriv)

# Manual Convolution
def manual_convolution(image, kernel):
    i_h, i_w = image.shape
    k_h, k_w = kernel.shape
    pad_h = k_h // 2
    pad_w = k_w // 2
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output = np.zeros_like(image)
    for i in range(i_h):
        for j in range(i_w):
            region = padded_image[i:i + k_h, j:j + k_w]
            output[i, j] = np.sum( kernel * region)
    return output

# GoG Filter Application
Ix = manual_convolution(image, Gx_deriv)
Iy = manual_convolution(image, Gy_deriv)

# Calculate Gradient Magnitude
grad_magnitude = np.sqrt(Ix**2 + Iy**2)

# SAVE RESULTS
plt.imsave("1_original_image.png", image_array, cmap='gray')
plt.imsave("2_enhanced_grayscale.png", enhanced_grayscale_array_rounded, cmap='gray')
plt.imsave("3_gradient_Ix.png", Ix, cmap='gray')
plt.imsave("4_gradient_Iy.png", Iy, cmap='gray')
plt.imsave("5_gradient_magnitude.png", grad_magnitude, cmap='gray')

# Auto-Correlation Matrix
MIx = Ix * Ix
MIy = Iy * Iy
MIxy = Ix * Iy

def create_uniform_kernel(size):
    return np.ones((size, size)) / (size * size)

kernel_5x5 = create_uniform_kernel(5)

# Do convolution using kernel 5 x 5
S_Ix2 = manual_convolution(MIx, kernel_5x5)
S_Iy2 = manual_convolution(MIy, kernel_5x5)
S_Ixy = manual_convolution(MIxy, kernel_5x5)

# Calculate Determinant and Trace from Matrix M
det_M = S_Ix2 * S_Iy2 - S_Ixy**2
trace_M = S_Ix2 + S_Iy2
eps = 1e-12

# Step 3: Calculate Förstner
k = 0.04
W_forstner = det_M / (trace_M + eps)
Q_forstner = (4 * det_M) / (trace_M**2 + eps)

# Step 4: Thresholding Corner detection (raw)
threshold_forstner = 0.004
corner_mask_forstner = (W_forstner > threshold_forstner) & (Q_forstner > 0.5)

# SAVE RESULTS
plt.imsave("6_S_Ix.png", S_Ix2, cmap='gray')
plt.imsave("7_S_Iy2.png", S_Iy2, cmap='gray')
plt.imsave("8_S_IxIy.png", S_Ixy, cmap='gray')
plt.imsave("9_W_forstner.png", W_forstner, cmap='jet')
plt.imsave("10_Q_forstner.png", Q_forstner, cmap='jet')

def corner_mask(W, Q, tw=0.005, tq=0.6):
    return (W > tw) & (Q > tq)

# Mask based on Forstner
Mc = corner_mask(W_forstner, Q_forstner, tw=0.004, tq=0.5)

# Load original image again
img_rgb_load = Image.open('ampelmaennchen.png').convert("RGB")
img_rgb_array = np.array(img_rgb_load, dtype=np.float32) / 255.0

# Create copy for final overlay
overlay_result = np.copy(img_rgb_array)
overlay_result[Mc] = [1.0, 0.0, 0.0]

plt.imsave("11_Corner_Mask_Mc.png", Mc.astype(np.uint8) * 255, cmap='gray')
plt.imsave("12_overlay_result.png", overlay_result)