import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def erosion(binary_array):                                                                       # Erosion
    rows, cols = binary_array.shape                                                              # Extract number of rows and columns
    eroded_array = np.zeros((rows, cols), dtype=np.bool)                                         # Create an identical 2d array full of 0s which will become the eroded image

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            patch = binary_array[i-2:i+2, j-2:j+2]                                               # Selecting a 4x4 matrix as a structuring element (SE)
            and_result = np.all(patch == 1)                                                      # ANDing of all 25 bits of input binary mask array
            eroded_array[i, j] = 1 if and_result else 0                                          # AND Value of pixel in output array corresponding to the origin

    return eroded_array                                                                          # Returning output new Eroded Array

def dilation(binary_array):                                                                      # Dilation
    rows, cols = binary_array.shape                                                              # Extract number of rows and columns
    dilated_array = np.zeros((rows, cols), dtype=np.bool)                                        # Create an identical 2d array full of 0s which will become the dilated image

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            patch = binary_array[i-2:i+2, j-2:j+2]                                               # Selecting a 4x4 matrix as a structuring element (SE)
            and_result = np.any(patch == 1)                                                      # ORing of all 25 bits of input binary mask array
            dilated_array[i, j] = 1 if and_result else 0                                         # OR Value of pixel in output array corresponding to the origin

    return dilated_array                                                                         # Returning output new Dilated Array

def opening(input_to_erosion):                                                                   # Closing = Dilation followed by Erosion
    input_to_dilation = erosion(input_to_erosion)                                                # Dilation process
    openned_array = dilation(input_to_dilation)                                                  # Erosion process
    return openned_array

def closing(input_to_dilation):                                                                  # Opening = Erosion followed by Dilation
    input_to_erosion = dilation(input_to_dilation)                                               # Dilation process
    closed_array = erosion(input_to_erosion)                                                     # Erosion process
    return closed_array

# 1. Prepare the image ------------------------------------------------------------------------------------------------------------------------------------------
img = Image.open('/Users/cristiad/Documents/Master Degree/1st Semester/Image Analysis/input_sat_image.jpg')          # Replace with your actual image file
image_array = np.array(img)                                                                      # Convert image to a 3 dimensional array

# 2. Convert RGB image to Grayscale image -----------------------------------------------------------------------------------------------------------------------
grayscale_array = image_array.mean(axis=2).astype(np.uint8)                                      # Compute mean across axis=2 i.e. across the color channels as 8 bit values
#print('Averaged array: ',grayscale_array)                                                       # Print out averaged array

# 3. Contrast Stretching -----------------------------------------------------------------------------------------------------------------------------------------
min_intensity = grayscale_array.min()                                                            # Find minimum grayscale value in avged image
max_intensity = grayscale_array.max()                                                            # Find maximum grayscale value in avged image
#print('min: ',min_intensity,'max: ',max_intensity)                                              # Check the min and max values

enhanced_grayscale_array = ((grayscale_array-min_intensity)/(max_intensity-min_intensity)) * 255 # Contrast Stretching
#print('Enhanced array: ',enhanced_grayscale_array)                                              # Print-out enhanced array

enhanced_grayscale_array_rounded = np.floor(enhanced_grayscale_array).astype(np.uint8)           # Flooring to 8 bit integer values
#print('Enhanced array rounded: ',enhanced_grayscale_array_rounded)                              # Print-out new enhanced array

# 4. Visualising Results -----------------------------------------------------------------------------------------------------------------------------------------
grayscale_img = Image.fromarray(grayscale_array, mode="L")                                       # Creating image from rgb to grayscale values
enhanced_grayscale_image = Image.fromarray(enhanced_grayscale_array_rounded, mode="L")           # Creating image from enhanced grayscale values

# grayscale_img.save("Grayscale Output.jpg")                                                       # Visualise initial image
# enhanced_grayscale_image.save("Enhanced Output.jpg")                                             # Visualise enhanced image
grayscale_img.show()
enhanced_grayscale_image.show()

# 5. Plotting Histograms -----------------------------------------------------------------------------------------------------------------------------------------
initial_grayscale_levels_falttened = grayscale_array.flatten()                                   # Flatten the 2D array into a 1D array of pixel intensities
enhanced_grayscale_levels_falttened = enhanced_grayscale_array_rounded.flatten()                 # Flatten the 2D array into a 1D array of pixel intensities

plt.hist(initial_grayscale_levels_falttened, bins=256, range=(0, 255), color='gray', edgecolor='black')
plt.title("Initial Image Histogram")
plt.xlabel("Gray Scale Values (0–255)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()                                                                                       # Visualise histogram of the initial image

plt.hist(enhanced_grayscale_levels_falttened, bins=256, range=(0, 255), color='gray', edgecolor='black')
plt.title("Enhanced Image Histogram")
plt.xlabel("Contrast Stretched Gray Scale Values (0–255)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()                                                                                       # Visualise histogram of the enhanced image

# 6. Pick a threshold = 91 ---------------------------------------------------------------------------------------------------------------------------------------
threshold_value = 91                                                                             # After analysing histogram of the enhanced image, threshold = 91
binary_mask_array = (enhanced_grayscale_array_rounded < threshold_value)                         # Since water area is area of interest, values below 91 are set to 1
binary_mask_image = Image.fromarray(binary_mask_array).convert("1")                              # Convert array to binary mask where each pixel is 1 or 0
# binary_mask_image.save("Thresholding Output.jpg")                                                # Visualise binary mask
binary_mask_image.show()

# 7. Opening, followed by Closing --------------------------------------------------------------------------------------------------------------------------------
o2c_opened_binary_mask_array = opening(binary_mask_array)                                        # Opening operation on Binary Mask
o2c_closed_binary_mask_array = closing(o2c_opened_binary_mask_array)                             # Closing operation on Opened Binary Mask
closed_image = Image.fromarray(o2c_closed_binary_mask_array).convert("1")                        # Convert array to binary mask where each pixel is 1 or 0
# closed_image.save("OpenClose Output.jpg")                                                        # Visualise Open + Close image
closed_image.show()

# 8. Closing, followed by Opening --------------------------------------------------------------------------------------------------------------------------------
c2o_closed_binary_mask_array = closing(binary_mask_array)                                        # Closing operation on Binary Mask
c2o_opened_binary_mask_array = opening(c2o_closed_binary_mask_array)                             # Opening operation on Opened Binary Mask
opened_image = Image.fromarray(c2o_opened_binary_mask_array).convert("1")                        # Convert array to binary mask where each pixel is 1 or 0
# opened_image.save("CloseOpen Output.jpg")                                                        # Visualise Close + Open image
opened_image.show()

# 9. Final Overlaying of Image -----------------------------------------------------------------------------------------------------------------------------------
rows, cols = enhanced_grayscale_array_rounded.shape
o2c_final_overlay_array = np.zeros((rows, cols), dtype=np.uint8)                                 # Create a blank 2d array that is completely black
c2o_final_overlay_array = np.zeros((rows, cols), dtype=np.uint8)                                 # Create a blank 2d array that is completely black
for i in range(1, rows - 1):
    for j in range(1, cols - 1):
        # Overlaying the non-water area in the black regions and keeping water region white
        o2c_final_overlay_array[i,j] = enhanced_grayscale_array_rounded[i,j] if o2c_closed_binary_mask_array[i,j] == 0 else 255
        c2o_final_overlay_array[i,j] = enhanced_grayscale_array_rounded[i,j] if c2o_opened_binary_mask_array[i,j] == 0 else 255

o2c_final_overlay_image = Image.fromarray(o2c_final_overlay_array, mode="L")                     # Convert array to 8bit greyscale image
c2o_final_overlay_image = Image.fromarray(c2o_final_overlay_array, mode="L")                     # Convert array to 8bit greyscale image
o2c_final_overlay_image.save("Opening + Closing - Final Overlay Output.jpg")                     # Visualise Image
c2o_final_overlay_image.save("Closing + Opening - Final Overlay Output.jpg")                     # Visualise Image