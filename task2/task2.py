import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 1. Prepare the image --------------------------------------------
image_pil = Image.open('/Users/cristiad/Documents/Master Degree/1st Semester/Image Analysis/iaor/task2/ampelmaennchen.png').convert("L")
image_array = np.array(image_pil, dtype=np.float32) / 255.0

# -------------------- GRAYSCALE & ENHANCEMENT --------------------
# Convert to Grayscale manually
grayscale_array = (image_array * 255).astype(np.uint8)


# Contrast stretching (enhancement)
min_intensity = grayscale_array.min()
max_intensity = grayscale_array.max()
enhanced_grayscale_array = ((grayscale_array - min_intensity) / (max_intensity - min_intensity)) * 255
enhanced_grayscale_array_rounded = np.floor(enhanced_grayscale_array).astype(np.uint8)

# Normalization
image = enhanced_grayscale_array_rounded.astype(np.float32) / 255.0

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

# -------------------- 5. SAVE RESULTS --------------------
plt.imsave("1_original_image.png", image_array, cmap='gray')
plt.imsave("2_enhanced_grayscale.png", enhanced_grayscale_array_rounded, cmap='gray')
plt.imsave("3_gradient_Ix.png", Ix, cmap='gray')
plt.imsave("4_gradient_Iy.png", Iy, cmap='gray')
plt.imsave("5_gradient_magnitude.png", grad_magnitude, cmap='gray')

# Optional: save all visualizations in one figure
plt.figure(figsize=(18, 4))
plt.subplot(1, 5, 1)
plt.imshow(image_array, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 5, 2)
plt.imshow(enhanced_grayscale_array_rounded, cmap='gray')
plt.title('Enhanced Grayscale')
plt.axis('off')

plt.subplot(1, 5, 3)
plt.imshow(Ix, cmap='gray')
plt.title('Gradient Ix')
plt.axis('off')

plt.subplot(1, 5, 4)
plt.imshow(Iy, cmap='gray')
plt.title('Gradient Iy')
plt.axis('off')

plt.subplot(1, 5, 5)
plt.imshow(grad_magnitude, cmap='gray')
plt.title('Gradient Magnitude')
plt.axis('off')

plt.tight_layout()
plt.savefig("all_results_combined.png", dpi=300)

print("All image printed out:")
print("- 1_original_image.png")
print("- 2_enhanced_grayscale.png")
print("- 3_gradient_Ix.png")
print("- 4_gradient_Iy.png")
print("- 5_gradient_magnitude.png")
print("- all_results_combined.png")