import numpy as np
import matplotlib.pyplot as plt
from skimage import data, filters, morphology, restoration, transform, registration, exposure
from scipy.ndimage import map_coordinates, binary_fill_holes
from skimage.io import imread
import cv2

image = imread('./duke_original/image/Subject_05_24.png', as_gray=True) # currently testing for a single image on the DUKE dataset, we can use a for loop for every image.
#TODO: Hepsini .png'ye çevirebilecek bir şey yaz.

# the preprocessing examples were done with the help of Par Kragsterman:
# https://about.cmrad.com/articles/the-ultimate-guide-to-preprocessing-medical-images-techniques-tools-and-best-practices-for-enhanced-diagnosis

def original_vs_preprocessed_plot(original, processed, title1= "Original Image", title2= "Preprocessed Image"):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title(title1)
    plt.imshow(original ,cmap= "gray")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title(title2)
    plt.imshow(processed, cmap= "gray")
    plt.axis("off")
    plt.show()

# threshold -> which pixels are considered the "background"
# np.max(image) -> basically the maximum intensity of the image. for example, if the threshold is 0.05, this tells
# us to remove the image parts with the intensity that is 5% of the maximum intensity.
def remove_background(image, threshold):
    mask = image > (threshold * np.max(image)) # pixels below the threshold are set to 0.
    mask = morphology.closing(mask, morphology.square(5)) # the closing algorithm fits gaps in the binary mask, removing small holes.
    mask = morphology.remove_small_objects(mask, min_size=500) # removes isolated small objects within the mask.
    return image * mask # the mask is applied here.

def remove_background_otsu(image): # uses otsu thresholding to remove the background.
    _, otsu_threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu_threshold

def plot_multiple_thresholds(image, thresholds):
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, len(thresholds)//2 + 1, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    for idx, threshold in enumerate(thresholds, start=2):
        bg_removed = remove_background(image, threshold)
        plt.subplot(2, len(thresholds)//2 + 1, idx)
        plt.imshow(bg_removed, cmap='gray')
        plt.title(f'Threshold = {threshold}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

threshold_values = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.28, 0.3, 0.5]


def denoise_image(image):
    return restoration.denoise_wavelet(image, method= "BayesShrink", mode="soft", rescale_sigma= True) #wavelet based denoising

denoised_image = denoise_image(image)
original_vs_preprocessed_plot(image, denoised_image, "Original Image", "Denoised Image Using Wavelets") # looks like not much happens. maybe this is not needed.


def resample_image(image, target_shape): # resampling = changing the pixel size of an image without altering its resolution.
    return transform.resize(image, target_shape, order=3, mode= "reflect", anti_aliasing= True)

resampled_image = resample_image(denoised_image, (250,500))
original_vs_preprocessed_plot(image, resampled_image, "Original Image", "Resampled Image") # looks like this will not be needed either. the size seems ideal but maybe
# we can ask çiğdem hoca.

# Image Registration: The process of aligning two images from different modalities or time points.
# I don't think this is needed in our application but here it is:

def register_images(fixed_image, moving_image):
    v, u = registration.optical_flow_tvl1(fixed_image, moving_image) # this created a displacement field, meaning a vector field that shows how each pixel should be shifted to match the fixed image.
    coords = np.meshgrid(np.arange(moving_image.shape[0]),
                         np.arange(moving_image.shape[1]),
                         indexing='ij')
    registered_image = map_coordinates(moving_image, 
                                       [coords[0] + v, coords[1] + u], 
                                       order=1)
    return registered_image

# test function to move the image.
def translate_image(image, tx=30, ty=20):
    moved_image = np.zeros_like(image)
    max_y, max_x = image.shape
    for y in range(max_y):
        for x in range(max_x):
            new_y, new_x = y + ty, x + tx
            if 0 <= new_y < max_y and 0 <= new_x < max_x:
                moved_image[new_y, new_x] = image[y, x]
    
    return moved_image

moved_image = translate_image(image)
registered_image = register_images(image, moved_image)
original_vs_preprocessed_plot(moved_image, registered_image, "Distorted Image", "Registered Image According to the Original Reference")


def normalize_intensity(image, min_percentile= 0.5, max_percentile=99.5): # normalize the image intensities to ensure consistency across the dataset.
    min_val = np.percentile(image, min_percentile)
    max_val = np.percentile(image, max_percentile)
    normalized_img = (image - min_val) / (max_val - min_val)
    return np.clip(normalized_img, 0, 1) # further ensures that no possible out-of-range values are present.

normalized_image = normalize_intensity(image)
original_vs_preprocessed_plot(image, normalized_image, "Original Image", "Normalized Image") # looks like this is the only method that worked.

plot_multiple_thresholds(normalized_image, threshold_values) # the thresholds definitely matter, however the closing and remove small objects operations didn't do much.

otsu_thresholded_image = remove_background_otsu(np.uint8(cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)))
original_vs_preprocessed_plot(image, otsu_thresholded_image, "Original Image", "Image with Otsu Thresholding")


def adapted_thresholding(image): # uses adapted thresholding to remove the background.
    smoothed_image = cv2.blur(image, (49, 49))
    bw_image = np.where(image > smoothed_image, 255, 0).astype(np.uint8)
    return bw_image

smoothed_image = adapted_thresholding(np.uint8(cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)))
original_vs_preprocessed_plot(image, smoothed_image, "Original Image", "Image with Adaptive Thresholding")
